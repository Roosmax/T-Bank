from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from src.config import DataConfig, EnvConfig, RSSMConfig, TrainConfig
from src.data import RolloutDataset
from src.models import RSSMState, WorldModel
from src.utils import ensure_dir, set_global_seed

def compute_losses(
    model: WorldModel,
    obs: Tensor,          # (B, L, 3, 64, 64)
    prev_action: Tensor,  # (B, L, A)
    reward: Tensor,       # (B, L)
    mask: Tensor,         # (B, L) 
    cfg: TrainConfig,
) -> Dict[str, Tensor]:

    b, length = obs.shape[:2]
    device = obs.device

    embed = model.encoder(obs.flatten(0, 1)).unflatten(0, (b, length))

    init = model.rssm.initial_state(b, device)
    post, prior = model.rssm.observe(embed, prev_action, init)

    feat = post.feat() 

    recon = model.decoder(feat.flatten(0, 1)).unflatten(0, (b, length))
    recon_err = ((recon - obs) ** 2).sum(dim=(-3, -2, -1))
    recon_loss = (recon_err * mask).sum() / mask.sum()
   
    post_d = Normal(post.mean, post.std)
    prior_d = Normal(prior.mean, prior.std)
    post_sg = Normal(post.mean.detach(), post.std.detach())
    prior_sg = Normal(prior.mean.detach(), prior.std.detach())

    kl_train_prior = kl_divergence(post_sg, prior_d).sum(-1)
    kl_train_post = kl_divergence(post_d, prior_sg).sum(-1)

    kl_balanced = (
        cfg.kl_balance * torch.clamp(kl_train_prior, min=cfg.free_nats)
        + (1.0 - cfg.kl_balance) * torch.clamp(kl_train_post, min=cfg.free_nats)
    )
    kl_loss = (kl_balanced * mask).sum() / mask.sum()
 
    kl = kl_divergence(post_d, prior_d).sum(-1)

    overshoot_loss = torch.zeros((), device=device)
    if cfg.overshoot_scale > 0 and cfg.overshoot_distance >= 2:
       
        rolled = RSSMState(*(f[:, 1:] for f in prior))
        num_terms = 0
        for d in range(2, cfg.overshoot_distance + 1):
          
            rolled = RSSMState(*(f[:, :-1] for f in rolled))
            length_d = rolled.deter.shape[1]  # = L - d
            if length_d <= 0:
                break
           
            act = prev_action[:, d:].flatten(0, 1)
            flat = RSSMState(*(f.flatten(0, 1) for f in rolled))
            nxt = model.rssm.prior_step(flat, act)
            rolled = RSSMState(*(f.unflatten(0, (b, length_d)) for f in nxt))

            kl_d = kl_divergence(
                Normal(post.mean[:, d:].detach(), post.std[:, d:].detach()),
                Normal(rolled.mean, rolled.std),
            ).sum(-1)  # (B, L-d)
            kl_d = torch.clamp(kl_d, min=cfg.free_nats)
            m = mask[:, d:]
            overshoot_loss = overshoot_loss + (kl_d * m).sum() / m.sum()
            num_terms += 1
        if num_terms > 0:
            overshoot_loss = overshoot_loss / num_terms 

    reward_pred = model.reward_head(feat.flatten(0, 1)).unflatten(0, (b, length))
    r_mask = mask.clone()
    r_mask[:, 0] = 0.0
    reward_loss = (((reward_pred - reward) ** 2) * r_mask).sum() / r_mask.sum()

    total = (
        recon_loss
        + cfg.kl_scale * kl_loss
        + cfg.overshoot_scale * overshoot_loss
        + cfg.reward_scale * reward_loss
    )
    return {
        "total": total,
        "recon": recon_loss,
        "kl": (kl * mask).sum() / mask.sum(),
        "overshoot": overshoot_loss,
        "reward": reward_loss,
    }

def train(train_cfg: TrainConfig) -> None:
    env_cfg, data_cfg, rssm_cfg = EnvConfig(), DataConfig(), RSSMConfig()
    set_global_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = RolloutDataset(
        data_cfg.out_path, train_cfg.seq_len, env_cfg.num_actions
    )
    print(
        f"dataset: {dataset.num_episodes} episodes, "
        f"{len(dataset.actions)} transitions"
    )

    model = WorldModel(rssm_cfg, action_dim=env_cfg.num_actions).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"world model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    rng = np.random.default_rng(train_cfg.seed)

    history: Dict[str, List[float]] = {
        k: [] for k in ("total", "recon", "kl", "overshoot", "reward")
    }
    t0 = time.time()
    for step in range(1, train_cfg.train_steps + 1):
        batch = dataset.sample_batch(train_cfg.batch_size, rng)

        obs = WorldModel.preprocess(
            torch.from_numpy(batch.obs).to(device, non_blocking=True)
        )
        prev_action = torch.from_numpy(batch.prev_action).to(device)
        reward = torch.from_numpy(batch.reward).to(device)
        mask = torch.from_numpy(batch.mask).to(device)

        losses = compute_losses(model, obs, prev_action, reward, mask, train_cfg)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
     
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        for key in history:
            history[key].append(float(losses[key].detach()))

        if step % train_cfg.log_every == 0:
            avg = {k: np.mean(v[-train_cfg.log_every:]) for k, v in history.items()}
            print(
                f"step {step:5d} | total {avg['total']:9.1f} | "
                f"recon {avg['recon']:9.1f} | kl {avg['kl']:6.2f} | "
                f"overshoot {avg['overshoot']:6.2f} | "
                f"reward {avg['reward']:.4f} | {time.time() - t0:5.0f}s"
            )

        if step % 1000 == 0 or step == train_cfg.train_steps:
            ensure_dir(train_cfg.ckpt_path)
            torch.save(
                {
                    "model": model.state_dict(),
                    "rssm_cfg": asdict(rssm_cfg),
                    "env_cfg": asdict(env_cfg),
                    "train_cfg": asdict(train_cfg),
                    "step": step,
                },
                train_cfg.ckpt_path,
            )

    print(f"checkpoint saved to {train_cfg.ckpt_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    for ax, key in zip(axes, ("total", "recon", "kl", "overshoot", "reward")):
        ax.plot(history[key])
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.set_yscale("log")
    fig.tight_layout()
    ensure_dir(train_cfg.curves_path)
    fig.savefig(train_cfg.curves_path, dpi=120)
    print(f"loss curves saved to {train_cfg.curves_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RSSM world model.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.steps is not None:
        cfg.train_steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    train(cfg)

if __name__ == "__main__":
    main()
