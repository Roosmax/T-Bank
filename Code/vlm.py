from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import DataConfig, EnvConfig, TrainConfig, VLMConfig
from src.utils import ensure_dir

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def build_palette(
    frames_uint8: np.ndarray, min_frac: float = 0.002
) -> torch.Tensor:

    colors, counts = np.unique(
        frames_uint8.reshape(-1, 3), axis=0, return_counts=True
    )
    keep = counts / counts.sum() >= min_frac
    return torch.from_numpy(colors[keep].astype(np.float32)) / 255.0


def project_to_palette(frames: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    
    n, _, h, w = frames.shape
    pal = palette.to(frames.device)
    x = (frames + 0.5).clamp(0.0, 1.0).movedim(1, -1).reshape(-1, 3)  # (M, 3)

    out = torch.empty_like(x)
    chunk = 4_000_000 
    for s in range(0, x.shape[0], chunk):
        xs = x[s : s + chunk]
       
        d = torch.cdist(xs.unsqueeze(0), pal.unsqueeze(0)).squeeze(0)
        out[s : s + chunk] = pal[d.argmin(dim=1)]

    return out.reshape(n, h, w, 3).movedim(-1, 1) - 0.5


class CLIPScorer:
   
    def __init__(self, cfg: VLMConfig, device: torch.device) -> None:
        import open_clip

        self.cfg = cfg
        self.device = device

        self.model, _, _ = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained
        )
        self.model = self.model.to(device).eval()
        
        self.fp16 = cfg.use_fp16 and device.type == "cuda"
        if self.fp16:
            self.model = self.model.half()
        
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._tokenizer = open_clip.get_tokenizer(cfg.model_name)

        goals = self.encode_texts([g for g, _ in cfg.prompt_pairs])
        bases = self.encode_texts([b for _, b in cfg.prompt_pairs])
        self.score_vec = (goals - bases).mean(dim=0)

        self.image_size: int = self.model.visual.image_size[0] \
            if isinstance(self.model.visual.image_size, (tuple, list)) \
            else self.model.visual.image_size

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_texts(self, prompts: List[str]) -> Tensor:
        tokens = self._tokenizer(prompts).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)

    @torch.no_grad()
    def embed_images_chw(self, frames: Tensor) -> Tensor:
    
        embs: List[Tensor] = []
        for start in range(0, frames.shape[0], self.cfg.batch_size):
            x = frames[start : start + self.cfg.batch_size].to(self.device)

            kwargs = {} if self.cfg.upscale_mode == "nearest" else {
                "align_corners": False
            }
            x = F.interpolate(
                x, size=self.image_size, mode=self.cfg.upscale_mode, **kwargs
            ).clamp(0.0, 1.0)
    
            mean = x.new_tensor(_CLIP_MEAN).view(1, 3, 1, 1)
            std = x.new_tensor(_CLIP_STD).view(1, 3, 1, 1)
            x = (x - mean) / std

            if self.fp16:
                x = x.half()
           
            embs.append(F.normalize(self.model.encode_image(x).float(), dim=-1))
        return torch.cat(embs)

    @torch.no_grad()
    def score_chw(self, frames: Tensor) -> Tensor:
        return self.embed_images_chw(frames) @ self.score_vec  # (N,)

    @torch.no_grad()
    def embed_uint8(self, frames: np.ndarray) -> Tensor:

        x = torch.from_numpy(frames).to(self.device)
        x = x.float().movedim(-1, -3) / 255.0 - 0.5
        return self.embed_images_chw(x)

    @torch.no_grad()
    def score_uint8(self, frames: np.ndarray) -> np.ndarray:
        return (self.embed_uint8(frames) @ self.score_vec).cpu().numpy()

def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:

    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def collect_labeled_frames(
    data_path: str, seq_len: int, num_actions: int, num_negatives: int = 500
) -> tuple[np.ndarray, np.ndarray]:
  
    from src.data import RolloutDataset

    dataset = RolloutDataset(data_path, seq_len, num_actions)
    succ = dataset.success_episodes()

    pos_frames = np.stack(
        [dataset.get_episode(int(e))[0][-1] for e in succ]
    ) 

    rng = np.random.default_rng(0)
    neg_frames = []
    while len(neg_frames) < num_negatives:
        ep = int(rng.integers(dataset.num_episodes))
        frames, _, rewards = dataset.get_episode(ep)
        t = int(rng.integers(len(frames)))
        if rewards[-1] > 0 and t == len(frames) - 1:
            continue
        neg_frames.append(frames[t])
    return pos_frames, np.stack(neg_frames)

def calibrate(
    scorer: CLIPScorer,
    data_path: str,
    seq_len: int,
    num_actions: int,
    num_negatives: int = 500,
    hist_path: str = "outputs/vlm_calibration.png",
) -> float:

    pos_frames, neg_frames = collect_labeled_frames(
        data_path, seq_len, num_actions, num_negatives
    )
    pos_scores = scorer.score_uint8(pos_frames)
    neg_scores = scorer.score_uint8(neg_frames)
    auc = roc_auc(pos_scores, neg_scores)

    print(f"positives (agent on goal): {len(pos_scores)} frames | "
          f"mean score {pos_scores.mean():+.4f}")
    print(f"negatives (anywhere else): {len(neg_scores)} frames | "
          f"mean score {neg_scores.mean():+.4f}")
    print(f"ROC-AUC: {auc:.3f}   (0.5 = blind, 1.0 = perfect)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(neg_scores.min(), pos_scores.min()),
        max(neg_scores.max(), pos_scores.max()),
        40,
    )
    ax.hist(neg_scores, bins=bins, alpha=0.6, density=True,
            label="negatives (agent elsewhere)")
    ax.hist(pos_scores, bins=bins, alpha=0.6, density=True,
            label="positives (agent on goal)")
    ax.set_xlabel("CLIP contrastive score: cos(goal) -- cos(baseline)")
    ax.set_ylabel("density")
    ax.set_title(f"VLM scorer calibration -- ROC-AUC = {auc:.3f}")
    ax.legend()
    fig.tight_layout()
    ensure_dir(hist_path)
    fig.savefig(hist_path, dpi=120)
    print(f"histogram saved to {hist_path}")
    return auc

GOAL_CANDIDATES: List[str] = [
    "a red arrow standing on the green square",
    "a red triangle on top of a green square",
    "the red agent has reached the green goal",
    "a red arrow inside a green square",
    "a grid world where the red triangle overlaps the green square",
    "the red player standing on the green tile",
    "only a green square with a red triangle inside it",
    "the red triangle and the green square merged together",
    "a red triangle standing on the green tile in the corner",
    "the red triangle touching the green square",
]
BASELINE_CANDIDATES: List[str] = [
    "a red arrow far away from the green square",
    "a red triangle and a separate green square",
    "the red agent has not reached the green goal",
    "a red arrow outside a green square",
    "a grid world with a red triangle and a green square in different places",
    "an empty dark room",
    "the red triangle and the green square apart from each other",
    "a red triangle alone in a dark grid",
    "an empty grid with a green square in the corner",
]

def search_prompts(
    scorer: CLIPScorer, data_path: str, seq_len: int, num_actions: int
) -> None:

    pos_frames, neg_frames = collect_labeled_frames(
        data_path, seq_len, num_actions
    )
    pos_emb = scorer.embed_uint8(pos_frames)  # (P, 512)
    neg_emb = scorer.embed_uint8(neg_frames)  # (N, 512)

    results = []
    for goal in GOAL_CANDIDATES:
        for base in BASELINE_CANDIDATES:
            text = scorer.encode_texts([goal, base])  # (2, 512)
            pos = (pos_emb @ text.T)[:, 0] - (pos_emb @ text.T)[:, 1]
            neg = (neg_emb @ text.T)[:, 0] - (neg_emb @ text.T)[:, 1]
            auc = roc_auc(pos.cpu().numpy(), neg.cpu().numpy())
            results.append((auc, goal, base))

    results.sort(reverse=True)
    print(f"{'AUC':>6}  goal || baseline")
    print("-" * 100)
    for auc, goal, base in results[:15]:
        print(f"{auc:6.3f}  '{goal}'  ||  '{base}'")
    print("...")
    for auc, goal, base in results[-3:]:
        print(f"{auc:6.3f}  '{goal}'  ||  '{base}'")

def _plan_reaches_goal(actions: np.ndarray) -> bool:

    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    x, y, d = 1, 1, 0
    for a in actions:
        if a == 0:
            d = (d - 1) % 4
        elif a == 1:
            d = (d + 1) % 4
        else:
            nx, ny = x + dirs[d][0], y + dirs[d][1]
            if 1 <= nx <= 6 and 1 <= ny <= 6:
                x, y = nx, ny
        if (x, y) == (6, 6):
            return True
    return False

def search_prompts_imagined(vlm_cfg: VLMConfig, num_seeds: int = 5) -> None:
   
    import torch as _torch

    from src.config import (
        DataConfig, EnvConfig, PlannerConfig, TrainConfig
    )
    from src.env import make_env
    from src.models import RSSMState, WorldModel
    from src.visualize import load_world_model

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    env_cfg, data_cfg = EnvConfig(), DataConfig()
    train_cfg, plan_cfg = TrainConfig(), PlannerConfig()
    model = load_world_model(train_cfg.ckpt_path, device)
    palette = build_palette(np.load(data_cfg.out_path)["frames"][:2000])
    env = make_env(env_cfg)

    obs, _ = env.reset(seed=1000)
    with _torch.no_grad():
        o = WorldModel.preprocess(
            _torch.from_numpy(obs.copy()).to(device)
        ).unsqueeze(0)
        state0, _ = model.rssm.posterior_step(
            model.rssm.initial_state(1, device),
            _torch.zeros(1, env_cfg.num_actions, device=device),
            model.encoder(o),
        )

    scorer = CLIPScorer(vlm_cfg, device)
    n, h = plan_cfg.num_candidates, plan_cfg.horizon
    batches = []
    for seed in range(num_seeds):
        g = _torch.Generator(device=device.type).manual_seed(seed)
        acts = _torch.randint(0, 3, (n, h), device=device, generator=g)
        for t in range(1, h):
            rep = (
                _torch.rand(n, device=device, generator=g)
                < plan_cfg.action_repeat_prob
            )
            acts[:, t] = _torch.where(
                rep, acts[:, t - 1],
                _torch.randint(0, 3, (n,), device=device, generator=g),
            )
        onehot = _torch.zeros(n, h, 3, device=device)
        onehot.scatter_(2, acts.unsqueeze(-1), 1.0)
        with _torch.no_grad():
            traj = model.rssm.imagine(
                RSSMState(*(f.expand(n, -1) for f in state0)),
                onehot, sample=False,
            )
            frames = project_to_palette(
                model.decoder(traj.feat().flatten(0, 1)), palette
            )
            emb = scorer.embed_images_chw(frames)  # (N*H, 512)
        hits = np.array([
            _plan_reaches_goal(p) for p in acts.cpu().numpy()
        ])
        batches.append((emb, hits))
    print(f"prepared {num_seeds} plan batches "
          f"({sum(b[1].sum() for b in batches)} true goal plans total)")

    results = []
    for goal in GOAL_CANDIDATES:
        for base in BASELINE_CANDIDATES:
            text = scorer.encode_texts([goal, base])
            v = text[0] - text[1]
            wins = 0
            for emb, hits in batches:
                if not hits.any():
                    continue
                s = (emb @ v).view(n, h).max(dim=1).values.cpu().numpy()
                wins += bool(hits[int(s.argmax())])
            results.append((wins, goal, base))

    results.sort(key=lambda r: -r[0])
    print(f"\nwins/{num_seeds} | goal || baseline")
    for wins, goal, base in results[:10]:
        print(f"  {wins}/{num_seeds}   | '{goal}' || '{base}'")

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="VLM scorer calibration.")
    parser.add_argument("--search", action="store_true",
                        help="Rank candidate prompt pairs by AUC on REAL "
                             "frames instead of calibrating the configured "
                             "pair.")
    parser.add_argument("--imagined", action="store_true",
                        help="Rank prompt pairs in the planner regime "
                             "(imagined + palette-projected frames). Run "
                             "this after every world-model retrain.")
    args = parser.parse_args()

    env_cfg, data_cfg, train_cfg, vlm_cfg = (
        EnvConfig(), DataConfig(), TrainConfig(), VLMConfig()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading CLIP {vlm_cfg.model_name} ({vlm_cfg.pretrained}) on {device}...")

    if args.imagined:
        search_prompts_imagined(vlm_cfg)
        return

    scorer = CLIPScorer(vlm_cfg, device)
    if args.search:
        search_prompts(
            scorer, data_cfg.out_path, train_cfg.seq_len, env_cfg.num_actions
        )
    else:
        print("prompt ensemble (goal || baseline):")
        for goal, base in vlm_cfg.prompt_pairs:
            print(f"  '{goal}'  ||  '{base}'")
        print()
        calibrate(
            scorer, data_cfg.out_path, train_cfg.seq_len, env_cfg.num_actions
        )

if __name__ == "__main__":
    main()
