from __future__ import annotations

import os
import random
from typing import Sequence
import numpy as np
from PIL import Image

def set_global_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> None:

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_image_grid(
    frames: Sequence[np.ndarray],
    nrow: int = 4,
    upscale: int = 2,
) -> Image.Image:

    frames = list(frames)
    assert len(frames) > 0, "No frames given"
    h, w, _ = frames[0].shape
    ncol = nrow
    nrows = (len(frames) + ncol - 1) // ncol

    pad = 2
    canvas = np.full(
        (nrows * (h + pad) - pad, ncol * (w + pad) - pad, 3),
        255,
        dtype=np.uint8,
    )
    for i, frame in enumerate(frames):
        r, c = divmod(i, ncol)
        y, x = r * (h + pad), c * (w + pad)
        canvas[y : y + h, x : x + w] = frame

    img = Image.fromarray(canvas)
    if upscale > 1:
        img = img.resize(
            (img.width * upscale, img.height * upscale),
            resample=Image.NEAREST,
        )
    return img
