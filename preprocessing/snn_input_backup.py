
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import (
    train_loader,
    validation_loader,
    test_loader,
    CLASS_NAMES,
    NUM_CLASSES,
    DEVICE,
)

TIME_STEPS = 10

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


def rate_encode(images, time_steps=TIME_STEPS):
    """
    Deterministic rate encoding.

    Input:  [B, C, H, W]
    Output: [T, B, C, H, W]

    Values remain in [0, 1]. Repeating the rate across
    time avoids introducing random noise into each image.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor.")

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images with shape [B, 3, H, W].")

    if not isinstance(time_steps, int) or time_steps < 1:
        raise ValueError("time_steps must be a positive integer.")

    mean = IMAGENET_MEAN.to(
        device=images.device, dtype=images.dtype
    )
    std = IMAGENET_STD.to(
        device=images.device, dtype=images.dtype
    )

    pixel_values = (images * std + mean).clamp(0.0, 1.0)

    # Keep the same image information at every simulation step.
    rates = pixel_values.to(torch.float32)
    spikes = rates.unsqueeze(0).expand(
        time_steps, -1, -1, -1, -1
    ).contiguous()

    return spikes


def convert_batch_to_spikes(
    images, labels=None, time_steps=TIME_STEPS
):
    spikes = rate_encode(images, time_steps=time_steps)

    if labels is None:
        return spikes

    return spikes, labels


if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    images = images[:2].to(DEVICE)
    labels = labels[:2].to(DEVICE)

    spikes, labels = convert_batch_to_spikes(
        images, labels, TIME_STEPS
    )

    print("Device:", DEVICE)
    print("Classes:", NUM_CLASSES)
    print("Class names:", len(CLASS_NAMES))
    print("Image shape:", tuple(images.shape))
    print("Encoded shape:", tuple(spikes.shape))
    print("Values within [0, 1]:", bool(
        ((spikes >= 0) & (spikes <= 1)).all().item()
    ))
    print("Same rates across time:", bool(
        torch.equal(spikes[0], spikes[-1])
    ))
    print("Labels:", labels.tolist())
    print("Encoder test completed.")
