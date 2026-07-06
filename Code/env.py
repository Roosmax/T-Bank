from __future__ import annotations
from typing import Optional
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from src.config import EnvConfig

class ReducedActionWrapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env, num_actions: int = 3) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(num_actions)

    def action(self, action: int) -> int:

        return action


def make_env(cfg: EnvConfig, render_mode: Optional[str] = None) -> gym.Env:
 
    env = gym.make(
        cfg.env_id,
        max_steps=cfg.max_steps,
        render_mode=render_mode,
        highlight=False,
    )

    env = RGBImgObsWrapper(env, tile_size=cfg.tile_size)
    env = ImgObsWrapper(env)
    env = ReducedActionWrapper(env, num_actions=cfg.num_actions)

    obs_shape = env.observation_space.shape
    expected = (cfg.image_size, cfg.image_size, 3)
    assert obs_shape == expected, (
        f"Unexpected observation shape {obs_shape}, expected {expected}. "
        "Check tile_size / env layout."
    )
    return env
