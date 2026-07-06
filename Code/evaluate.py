from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import argparse
import torch
import json
import time
import os

from typing import Dict, List, Optional, Protocol

from src.config import (
    DataConfig, EnvConfig, EvalConfig, PlannerConfig, TrainConfig, VLMConfig
)
from src.env import make_env
from src.planner import (
    MPCAgent, RandomShootingPlanner, RewardTrajectoryScorer,
    VLMTrajectoryScorer,
)
from src.utils import ensure_dir
from src.visualize import load_world_model
from src.vlm import CLIPScorer, build_palette

class Agent(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> int: ...

class RandomAgent:
    def __init__(self, num_actions: int, rng: np.random.Generator) -> None:
        self.num_actions = num_actions
        self.rng = rng

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.num_actions))

def run_episode(
    env, agent: Agent, seed: int, record: bool
) -> Dict[str, object]:

    obs, _ = env.reset(seed=seed)
    agent.reset()
    frames: List[np.ndarray] = [obs.copy()] if record else []
    total, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        steps += 1
        if record:
            frames.append(obs.copy())
    return {
        "seed": seed,
        "success": total > 0.0,
        "return": total,
        "steps": steps,
        "frames": frames,
    }

def save_gif(frames: List[np.ndarray], path: str, cfg: EvalConfig) -> None:

    k = cfg.gif_upscale
    big = [np.repeat(np.repeat(f, k, axis=0), k, axis=1) for f in frames]

    big += [big[-1]] * 5
    ensure_dir(path)
    iio.imwrite(path, big, duration=cfg.gif_frame_ms, loop=0)

def build_agent(
    method: str,
    env_cfg: EnvConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> Agent:

    if method == "random":
        return RandomAgent(env_cfg.num_actions, rng)


    train_cfg, plan_cfg = TrainConfig(), PlannerConfig()
    model = load_world_model(train_cfg.ckpt_path, device)
    planner = RandomShootingPlanner(plan_cfg, env_cfg.num_actions)

    if method == "reward_mpc":
        scorer = RewardTrajectoryScorer()
    elif method == "vlm_mpc":
        clip_scorer = CLIPScorer(VLMConfig(), device)
        palette = build_palette(
            np.load(DataConfig().out_path)["frames"][:2000]
        )
        scorer = VLMTrajectoryScorer(clip_scorer, plan_cfg, palette=palette)
    else:
        raise ValueError(f"unknown method: {method}")
    return MPCAgent(model, planner, scorer, env_cfg.num_actions, device)

def evaluate(methods: List[str], eval_cfg: EvalConfig) -> None:
    env_cfg = EnvConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(env_cfg)

    all_results: Dict[str, List[Dict[str, object]]] = {}
    for method in methods:
        rng = np.random.default_rng(eval_cfg.seed)
        agent = build_agent(method, env_cfg, device, rng)
        results = []
        t0 = time.time()
        for ep in range(eval_cfg.num_episodes):
            ep_seed = eval_cfg.seed + ep

            torch.manual_seed(ep_seed)
            record = ep < eval_cfg.gifs_per_method
            res = run_episode(env, agent, ep_seed, record)
            if record:
                path = os.path.join(
                    eval_cfg.gif_dir, f"{method}_seed{ep_seed}.gif"
                )
                save_gif(res["frames"], path, eval_cfg)
            res.pop("frames")
            results.append(res)
            print(
                f"  {method} ep {ep:2d} (seed {ep_seed}): "
                f"{'SUCCESS' if res['success'] else 'failure':7s} | "
                f"steps {res['steps']:3d} | return {res['return']:.3f}"
            )
        elapsed = time.time() - t0
        all_results[method] = results
        print(f"{method}: done in {elapsed:.0f}s\n")

    print("=" * 68)
    print(f"{'method':12s} | {'success rate':>12s} | {'mean return':>11s} | "
          f"{'mean steps':>10s}")
    print("-" * 68)
    summary: Dict[str, Dict[str, float]] = {}
    for method, results in all_results.items():
        succ = float(np.mean([r["success"] for r in results]))
        ret = float(np.mean([r["return"] for r in results]))
        steps = float(np.mean([r["steps"] for r in results]))
        summary[method] = {
            "success_rate": succ, "mean_return": ret, "mean_steps": steps,
        }
        print(f"{method:12s} | {succ:12.0%} | {ret:11.3f} | {steps:10.1f}")
    print("=" * 68)

    ensure_dir(eval_cfg.results_path)
    with open(eval_cfg.results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "num_episodes": eval_cfg.num_episodes,
                    "seed": eval_cfg.seed,
                },
                "summary": summary,
                "episodes": all_results,
            },
            f, indent=2,
        )
    print(f"results saved to {eval_cfg.results_path}")
    print(f"GIFs saved to {eval_cfg.gif_dir}/")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the three agents.")
    parser.add_argument("--methods", nargs="*",
                        default=["random", "reward_mpc", "vlm_mpc"],
                        choices=["random", "reward_mpc", "vlm_mpc"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = EvalConfig()
    if args.episodes is not None:
        cfg.num_episodes = args.episodes
    if args.seed is not None:
        cfg.seed = args.seed
    evaluate(args.methods, cfg)

if __name__ == "__main__":
    main()
