from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List
import numpy as np

from src.config import DataConfig, EnvConfig
from src.env import make_env
from src.utils import ensure_dir, make_image_grid

@dataclass
class Episode:

    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    @property
    def success(self) -> bool:
        # MiniGrid gives reward = 1 - 0.9 * (step / max_steps) > 0 only on
        # reaching the goal; every other step has reward exactly 0
        return bool(self.rewards[-1] > 0)

def biased_random_action(
    rng: np.random.Generator, num_actions: int, forward_bias: float
) -> int:

    p_turn = (1.0 - forward_bias) / 2.0
    return int(rng.choice(num_actions, p=[p_turn, p_turn, forward_bias]))

def collect_episode(env, rng: np.random.Generator, cfg: DataConfig, seed: int) -> Episode:
    frames: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []

    obs, _ = env.reset(seed=seed)
    frames.append(obs)

    terminated = truncated = False
    while not (terminated or truncated):
        act = biased_random_action(rng, env.action_space.n, cfg.forward_bias)
        obs, reward, terminated, truncated, _ = env.step(act)
        frames.append(obs)
        actions.append(act)
        rewards.append(float(reward))

    return Episode(
        frames=np.asarray(frames, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
    )

def save_dataset(episodes: List[Episode], out_path: str) -> None:
    ensure_dir(out_path)
    np.savez_compressed(
        out_path,
        frames=np.concatenate([ep.frames for ep in episodes], axis=0),
        actions=np.concatenate([ep.actions for ep in episodes], axis=0),
        rewards=np.concatenate([ep.rewards for ep in episodes], axis=0),
        episode_lengths=np.asarray([len(ep.actions) for ep in episodes], dtype=np.int64),
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect random rollouts in MiniGrid.")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--seed", type=int, default=None, help="Override base seed.")
    parser.add_argument("--out", type=str, default=None, help="Override output .npz path.")
    args = parser.parse_args()

    env_cfg = EnvConfig()
    data_cfg = DataConfig()
    if args.episodes is not None:
        data_cfg.num_episodes = args.episodes
    if args.seed is not None:
        data_cfg.seed = args.seed
    if args.out is not None:
        data_cfg.out_path = args.out

    env = make_env(env_cfg)
    rng = np.random.default_rng(data_cfg.seed)

    episodes: List[Episode] = []
    for ep_idx in range(data_cfg.num_episodes):
        ep = collect_episode(env, rng, data_cfg, seed=data_cfg.seed + ep_idx)
        episodes.append(ep)
        if (ep_idx + 1) % 50 == 0:
            print(f"  collected {ep_idx + 1}/{data_cfg.num_episodes} episodes")

    env.close()
    save_dataset(episodes, data_cfg.out_path)

    lengths = np.asarray([len(ep.actions) for ep in episodes])
    successes = sum(ep.success for ep in episodes)
    total_steps = int(lengths.sum())
    print("\nDataset summary")
    print(f"episodes        : {len(episodes)}")
    print(f"transitions     : {total_steps}")
    print(f"frames          : {total_steps + len(episodes)}")
    print(f"episode length  : mean {lengths.mean():.1f} | min {lengths.min()} | max {lengths.max()}")
    print(f"success episodes: {successes} ({100.0 * successes / len(episodes):.1f}%)")
    print(f"saved to        : {data_cfg.out_path}")

    # Visual sanity check
    all_frames = np.concatenate([ep.frames for ep in episodes], axis=0)
    picks = np.random.default_rng(0).choice(len(all_frames), size=16, replace=False)
    grid = make_image_grid([all_frames[i] for i in sorted(picks)], nrow=4, upscale=3)
    ensure_dir(data_cfg.preview_path)
    grid.save(data_cfg.preview_path)
    print(f"frame preview   : {data_cfg.preview_path}")

if __name__ == "__main__":
    main()
