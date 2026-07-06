from __future__ import annotations
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from src.config import RSSMConfig

class RSSMState(NamedTuple):

    deter: Tensor  # h_t (B, deter_dim)
    stoch: Tensor  # z_t sample (B, stoch_dim)
    mean: Tensor   # mu of z_t (B, stoch_dim)
    std: Tensor    # sigma of z_t (B, stoch_dim)

    def feat(self) -> Tensor:

        return torch.cat([self.deter, self.stoch], dim=-1)

def stack_states(states: list[RSSMState], dim: int = 1) -> RSSMState:

    return RSSMState(*(torch.stack(field, dim=dim) for field in zip(*states)))

class ConvEncoder(nn.Module):

    def __init__(self, embed_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ELU(),
        )
        assert embed_dim == 1024, "Conv stack output is hard-wired to 1024."

    def forward(self, obs: Tensor) -> Tensor:
        # (N, 3, 64, 64) float in [-0.5, 0.5] -> (N, 1024)
        return self.net(obs).flatten(start_dim=1)

class ConvDecoder(nn.Module):
  
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1024)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.ELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
        )

    def forward(self, feat: Tensor) -> Tensor:
        # (N, feat_dim) -> (N, 3, 64, 64)
        x = self.fc(feat)
        x = x.view(-1, 1024, 1, 1)
        return self.net(x)

class RewardHead(nn.Module):

    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: Tensor) -> Tensor:
        # (N, feat_dim) -> (N,)
        return self.net(feat).squeeze(-1)

class RSSM(nn.Module):

    def __init__(self, cfg: RSSMConfig, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim

        self.pre_gru = nn.Sequential(
            nn.Linear(cfg.stoch_dim + action_dim, cfg.hidden_dim), nn.ELU()
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.deter_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(cfg.deter_dim, cfg.hidden_dim), nn.ELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim),
        )

        self.post_net = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.embed_dim, cfg.hidden_dim), nn.ELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim),
        )

    @property
    def feat_dim(self) -> int:
        return self.cfg.deter_dim + self.cfg.stoch_dim

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        d, s = self.cfg.deter_dim, self.cfg.stoch_dim

        def zeros(n: int) -> Tensor:
            return torch.zeros(batch_size, n, device=device)

        return RSSMState(
            zeros(d), zeros(s), zeros(s), torch.ones(batch_size, s, device=device)
        )

    def _gaussian(self, raw: Tensor) -> Tuple[Tensor, Tensor]:

        mean, raw_std = torch.chunk(raw, 2, dim=-1)
        std = F.softplus(raw_std) + self.cfg.min_std
        return mean, std

    def prior_step(
        self, prev: RSSMState, prev_action: Tensor, sample: bool = True
    ) -> RSSMState:
    
        x = self.pre_gru(torch.cat([prev.stoch, prev_action], dim=-1))
        deter = self.gru(x, prev.deter)
        mean, std = self._gaussian(self.prior_net(deter))

        stoch = Normal(mean, std).rsample() if sample else mean
        return RSSMState(deter, stoch, mean, std)

    def posterior_step(
        self, prev: RSSMState, prev_action: Tensor, embed: Tensor
    ) -> Tuple[RSSMState, RSSMState]:

        prior = self.prior_step(prev, prev_action)
        mean, std = self._gaussian(
            self.post_net(torch.cat([prior.deter, embed], dim=-1))
        )
        stoch = Normal(mean, std).rsample()
        post = RSSMState(prior.deter, stoch, mean, std)  # h is shared.
        return post, prior

    def observe(
        self, embeds: Tensor, prev_actions: Tensor, init: RSSMState
    ) -> Tuple[RSSMState, RSSMState]:
        
        posts, priors = [], []
        state = init
        for t in range(embeds.shape[1]):  
            state, prior = self.posterior_step(
                state, prev_actions[:, t], embeds[:, t]
            )
            posts.append(state)
            priors.append(prior)
        return stack_states(posts), stack_states(priors)

    def imagine(
        self, init: RSSMState, actions: Tensor, sample: bool = True
    ) -> RSSMState:
        
        states = []
        state = init
        for t in range(actions.shape[1]):
            state = self.prior_step(state, actions[:, t], sample=sample)
            states.append(state)
        return stack_states(states)

class WorldModel(nn.Module):

    def __init__(self, cfg: RSSMConfig, action_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder(cfg.embed_dim)
        self.rssm = RSSM(cfg, action_dim)
        self.decoder = ConvDecoder(self.rssm.feat_dim)
        self.reward_head = RewardHead(self.rssm.feat_dim, cfg.hidden_dim)

    @staticmethod
    def preprocess(obs_uint8: Tensor) -> Tensor:
        obs = obs_uint8.float() / 255.0 - 0.5
        return obs.movedim(-1, -3)

    @staticmethod
    def postprocess(obs_pred: Tensor) -> Tensor:
        obs = (obs_pred.movedim(-3, -1) + 0.5).clamp(0.0, 1.0) * 255.0
        return obs.to(torch.uint8)
