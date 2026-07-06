from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class EnvConfig:
    env_id: str = "MiniGrid-Empty-8x8-v0"

    # Pixels rendered per grid cell. The full grid is 8x8 cells
    tile_size: int = 8
    image_size: int = 64  # grid_cells (8) * tile_size (8).

    # MiniGrid defines 7 actions, but only the first 3 matter in an empty room: 0=turn-left, 1=turn-right, 2=move-forward
    num_actions: int = 3
    max_steps: int = 60

@dataclass
class DataConfig:

    num_episodes: int = 500

    forward_bias: float = 0.5

    seed: int = 0  # Base seed

    out_path: str = "data/rollouts.npz"
    preview_path: str = "outputs/data_preview.png"

@dataclass
class RSSMConfig:

    deter_dim: int = 200  
    stoch_dim: int = 32   
    hidden_dim: int = 200 
    embed_dim: int = 1024  
    min_std: float = 0.1 

@dataclass
class TrainConfig:

    batch_size: int = 32
    seq_len: int = 25      
    
    train_steps: int = 10000
    lr: float = 3e-4
    grad_clip: float = 100.0 

    kl_scale: float = 1.0
    free_nats: float = 1.0 

    overshoot_distance: int = 8
    overshoot_scale: float = 0.0  

    kl_balance: float = 0.8
    reward_scale: float = 1.0

    seed: int = 0
    log_every: int = 100
    ckpt_path: str = "checkpoints/rssm.pt"
    curves_path: str = "outputs/train_curves.png"

@dataclass
class VLMConfig:


    model_name: str = "ViT-B-32-quickgelu"  

    pretrained: str = "openai"

  
    prompt_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("only a green square with a red triangle inside it",
         "a red triangle and a separate green square"),
    ])

    upscale_mode: str = "bicubic" 

    use_fp16: bool = True  
    
    batch_size: int = 512

@dataclass
class PlannerConfig:

    num_candidates: int = 1024 

    horizon: int = 15          

    action_repeat_prob: float = 0.7

    score_stride: int = 3     
    objective: str = "max"    

    time_penalty: float = 2.5e-4

@dataclass
class EvalConfig:
  
    num_episodes: int = 20
    seed: int = 5000

    results_path: str = "outputs/eval_results.json"
    gif_dir: str = "outputs/gifs"
    gifs_per_method: int = 3
    gif_upscale: int = 4      
    gif_frame_ms: int = 150  
