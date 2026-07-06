from __future__ import annotations

import argparse
import time
from typing import Optional, Protocol
import numpy as np
import torch
from torch import Tensor

from src.config import (
    DataConfig, EnvConfig, PlannerConfig, TrainConfig, VLMConfig
)
from src.env import make_env
from src.models import RSSMState, WorldModel
from src.vlm import CLIPScorer, build_palette, project_to_palette

class TrajectoryScorer(Protocol):
    def score(self, model: WorldModel, traj: RSSMState) -> Tensor:
        ...

class VLMTrajectoryScorer:

    def __init__(
        self,
        clip_scorer: CLIPScorer,
        cfg: PlannerConfig,
        palette: Optional[Tensor] = None,
    ) -> None:
        self.clip = clip_scorer
        self.cfg = cfg
        self.palette = palette

    @torch.no_grad()
    def score(self, model: WorldModel, traj: RSSMState) -> Tensor:
        n, horizon = traj.deter.shape[:2]

        ts = list(range(self.cfg.score_stride - 1, horizon,
                        self.cfg.score_stride))
        if ts[-1] != horizon - 1:
            ts.append(horizon - 1)

        feat = traj.feat()[:, ts]                      
        frames = model.decoder(feat.flatten(0, 1))     
        if self.palette is not None:
            frames = project_to_palette(frames, self.palette)
        scores = self.clip.score_chw(frames).view(n, len(ts))  # (N, K)

        scores = scores - self.cfg.time_penalty * scores.new_tensor(ts)

        if self.cfg.objective == "max":
            return scores.max(dim=1).values
       
        return scores.sum(dim=1)


class RewardTrajectoryScorer:
  
    def __init__(self, discount: float = 0.95) -> None:
        self.discount = discount

    @torch.no_grad()
    def score(self, model: WorldModel, traj: RSSMState) -> Tensor:
        n, horizon = traj.deter.shape[:2]
        rewards = model.reward_head(
            traj.feat().flatten(0, 1)
        ).view(n, horizon)
        weights = self.discount ** torch.arange(
            horizon, device=rewards.device, dtype=rewards.dtype
        )
        return (rewards * weights).sum(dim=1)

class RandomShootingPlanner:

    def __init__(self, cfg: PlannerConfig, num_actions: int) -> None:
        self.cfg = cfg
        self.num_actions = num_actions

    @torch.no_grad()
    def plan(
        self, model: WorldModel, state: RSSMState, scorer: TrajectoryScorer
    ) -> int:
     
        n, h, a = self.cfg.num_candidates, self.cfg.horizon, self.num_actions
        device = state.deter.device

        actions = torch.empty(n, h, dtype=torch.long, device=device)
        actions[:, 0] = torch.randint(0, a, (n,), device=device)
        for t in range(1, h):
            repeat = (
                torch.rand(n, device=device) < self.cfg.action_repeat_prob
            )
            resample = torch.randint(0, a, (n,), device=device)
            actions[:, t] = torch.where(repeat, actions[:, t - 1], resample)
        onehot = torch.zeros(n, h, a, device=device)
        onehot.scatter_(2, actions.unsqueeze(-1), 1.0)          

        start = RSSMState(
            *(field.expand(n, -1) for field in state)
        )
      
        traj = model.rssm.imagine(start, onehot, sample=False)

        scores = scorer.score(model, traj)
        best = int(scores.argmax())
        return int(actions[best, 0])

class MPCAgent:

    def __init__(
        self,
        model: WorldModel,
        planner: RandomShootingPlanner,
        scorer: TrajectoryScorer,
        num_actions: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.planner = planner
        self.scorer = scorer
        self.num_actions = num_actions
        self.device = device
        self._state: Optional[RSSMState] = None
        self._prev_action: Optional[Tensor] = None

    def reset(self) -> None:
    
        self._state = self.model.rssm.initial_state(1, self.device)
        self._prev_action = torch.zeros(1, self.num_actions, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        
        obs_t = torch.from_numpy(obs.copy()).to(self.device)
        obs_t = WorldModel.preprocess(obs_t).unsqueeze(0)   # (1, 3, 64, 64)
        embed = self.model.encoder(obs_t)                   # (1, 1024)

        self._state, _ = self.model.rssm.posterior_step(
            self._state, self._prev_action, embed
        )

        action = self.planner.plan(self.model, self._state, self.scorer)
        self._prev_action = torch.zeros_like(self._prev_action)
        self._prev_action[0, action] = 1.0
        return action