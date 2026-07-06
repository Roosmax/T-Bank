from __future__ import annotations
from typing import NamedTuple, Tuple
import numpy as np

class SequenceBatch(NamedTuple):

    obs: np.ndarray          # (B, L, 64, 64, 3)
    prev_action: np.ndarray  # (B, L, A)
    reward: np.ndarray       # (B, L)
    mask: np.ndarray         # (B, L)

class RolloutDataset:

    def __init__(self, path: str, seq_len: int, num_actions: int) -> None:
        data = np.load(path)
        self.frames: np.ndarray = data["frames"]      # (N_frames, 64, 64, 3)
        self.actions: np.ndarray = data["actions"]    # (N_steps,)
        self.rewards: np.ndarray = data["rewards"]    # (N_steps,)
        lengths: np.ndarray = data["episode_lengths"]  # (E,)

        self.seq_len = seq_len
        self.num_actions = num_actions
        self.num_episodes = len(lengths)

        self.ep_len = lengths
        self.frame_start = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]])
        self.act_start = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    def get_episode(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        t = int(self.ep_len[idx])
        fs, as_ = int(self.frame_start[idx]), int(self.act_start[idx])
        return (
            self.frames[fs : fs + t + 1],
            self.actions[as_ : as_ + t],
            self.rewards[as_ : as_ + t],
        )

    def success_episodes(self) -> np.ndarray:

        ends = self.act_start + self.ep_len - 1
        return np.nonzero(self.rewards[ends] > 0)[0]

    # Training batch sampling
    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
        success_frac: float = 0.3,
    ) -> SequenceBatch:
       
        L, A = self.seq_len, self.num_actions
        obs = np.zeros((batch_size, L, *self.frames.shape[1:]), dtype=np.uint8)
        prev_action = np.zeros((batch_size, L, A), dtype=np.float32)
        reward = np.zeros((batch_size, L), dtype=np.float32)
        mask = np.zeros((batch_size, L), dtype=np.float32)

        succ = self.success_episodes()
        for b in range(batch_size):
            force_success = succ.size > 0 and rng.random() < success_frac
            if force_success:
                ep = int(succ[rng.integers(len(succ))])
            else:
                ep = int(rng.integers(self.num_episodes))
            t_ep = int(self.ep_len[ep])          # actions
            n_frames = t_ep + 1                  # frames
            fs, as_ = int(self.frame_start[ep]), int(self.act_start[ep])

            n = min(L, n_frames)
            if force_success:

                s = n_frames - n
            else:

                s = int(rng.integers(0, n_frames - n + 1))

            obs[b, :n] = self.frames[fs + s : fs + s + n]
            mask[b, :n] = 1.0

            acts = self.actions[as_ + s : as_ + s + n - 1]     
            prev_action[b, np.arange(1, n), acts] = 1.0

            reward[b, 1:n] = self.rewards[as_ + s : as_ + s + n - 1]

            if n < L:
                obs[b, n:] = obs[b, n - 1]

        return SequenceBatch(obs, prev_action, reward, mask)
