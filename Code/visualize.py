from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import torch

from src.config import DataConfig, EnvConfig, RSSMConfig, TrainConfig
from src.data import RolloutDataset
from src.models import RSSMState, WorldModel
from src.utils import ensure_dir, make_image_grid


def load_world_model(
    ckpt_path: str, device: torch.device
) -> WorldModel:

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    rssm_cfg = RSSMConfig(**ckpt["rssm_cfg"])
    model = WorldModel(rssm_cfg, action_dim=ckpt["env_cfg"]["num_actions"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def to_onehot(actions: np.ndarray, num_actions: int, device: torch.device) -> torch.Tensor:

    onehot = torch.zeros(1, len(actions), num_actions, device=device)
    onehot[0, torch.arange(len(actions)), torch.as_tensor(actions)] = 1.0
    return onehot


@torch.no_grad()
def open_loop_prediction(
    model: WorldModel,
    frames: np.ndarray,
    actions: np.ndarray,
    context: int,
    horizon: int,
    device: torch.device,
    sample: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    num_actions = model.rssm.action_dim

    ctx_obs = WorldModel.preprocess(
        torch.from_numpy(frames[:context]).to(device)
    ).unsqueeze(0)
    embed = model.encoder(ctx_obs.flatten(0, 1)).unflatten(0, (1, context))

    prev_act = torch.zeros(1, context, num_actions, device=device)
    if context > 1:
        prev_act[:, 1:] = to_onehot(actions[: context - 1], num_actions, device)

    init = model.rssm.initial_state(1, device)
    post, _ = model.rssm.observe(embed, prev_act, init)
    recon = model.decoder(post.feat().flatten(0, 1))

    last = RSSMState(*(field[:, -1] for field in post))
    future_act = to_onehot(actions[context - 1 : context - 1 + horizon],
                           num_actions, device)
    imag = model.rssm.imagine(last, future_act, sample=sample)

    imag_frames = model.decoder(imag.feat().flatten(0, 1))

    return (
        WorldModel.postprocess(recon).cpu().numpy(),
        WorldModel.postprocess(imag_frames).cpu().numpy(),
    )

def visualize_episode(
    model: WorldModel,
    dataset: RolloutDataset,
    ep_idx: int,
    context: int,
    horizon: int,
    device: torch.device,
    out_path: str,
) -> None:

    frames, actions, _ = dataset.get_episode(ep_idx)
    horizon = min(horizon, len(actions) - context + 1)
    total = context + horizon

    recon, imag = open_loop_prediction(
        model, frames, actions, context, horizon, device
    )

    gt_row = [frames[t] for t in range(total)]

    model_row = [recon[t] for t in range(context)] + [imag[t] for t in range(horizon)]

    mse = [
        float(((imag[t].astype(np.float32) - frames[context + t].astype(np.float32)) ** 2).mean())
        for t in range(horizon)
    ]
    print(
        f"episode {ep_idx}: context={context}, horizon={horizon} | "
        f"imagination MSE per step: "
        + " ".join(f"{m:.0f}" for m in mse)
    )

    grid = make_image_grid(gt_row + model_row, nrow=total, upscale=2)
    ensure_dir(out_path)
    grid.save(out_path)
    print(f"  saved {out_path}  (row 1 = ground truth, row 2 = model; "
          f"model is open-loop from column {context + 1})")

@torch.no_grad()
def open_loop_stats(
    model: WorldModel,
    dataset: RolloutDataset,
    num_windows: int,
    context: int,
    horizon: int,
    device: torch.device,
    seed: int = 0,
    sample: bool = True,
) -> np.ndarray:
   
    rng = np.random.default_rng(seed)
    per_step: list[list[float]] = [[] for _ in range(horizon)]

    for _ in range(num_windows):
        ep = int(rng.integers(dataset.num_episodes))
        frames, actions, _ = dataset.get_episode(ep)

        if len(actions) < context + horizon:
            continue
        s = int(rng.integers(0, len(actions) - context - horizon + 1))
        _, imag = open_loop_prediction(
            model, frames[s:], actions[s:], context, horizon, device,
            sample=sample,
        )
        for t in range(horizon):
            gt = frames[s + context + t].astype(np.float32)
            per_step[t].append(float(((imag[t] - gt) ** 2).mean()))

    means = np.array([np.mean(v) if v else np.nan for v in per_step])
    print(f"\nopen-loop imagination MSE vs depth ({len(per_step[0])} windows):")
    for t, m in enumerate(means, start=1):
        print(f"  step {t:2d}: {m:7.1f}")
    return means

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise open-loop rollouts.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Episode indices; default picks one success + one other.")
    parser.add_argument("--context", type=int, default=5,
                        help="Frames given to the posterior before imagining.")
    parser.add_argument("--horizon", type=int, default=15,
                        help="Imagined steps (matches the MPC horizon regime).")
    parser.add_argument("--stats", type=int, default=0, metavar="N",
                        help="Also compute mean open-loop MSE over N random windows.")
    parser.add_argument("--mean", action="store_true",
                        help="Propagate prior means instead of sampling "
                             "(matches what the planner sees).")
    args = parser.parse_args()

    env_cfg, data_cfg, train_cfg = EnvConfig(), DataConfig(), TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_world_model(train_cfg.ckpt_path, device)
    dataset = RolloutDataset(data_cfg.out_path, train_cfg.seq_len, env_cfg.num_actions)

    episodes: Optional[List[int]] = args.episodes
    if not episodes:

        succ = dataset.success_episodes()
        long_enough = [
            int(e) for e in succ
            if dataset.ep_len[e] >= args.context + 5
        ]
        episodes = [long_enough[0] if long_enough else 0, 1]

    for ep in episodes:
        visualize_episode(
            model, dataset, ep, args.context, args.horizon, device,
            out_path=f"outputs/openloop_ep{ep}.png",
        )

    if args.stats > 0:
        open_loop_stats(
            model, dataset, args.stats, args.context, args.horizon, device,
            sample=not args.mean,
        )

if __name__ == "__main__":
    main()
