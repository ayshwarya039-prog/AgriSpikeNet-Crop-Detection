
import torch

TIME_STEPS = 10


def rate_encode(images, time_steps=TIME_STEPS):
    """
    Experimental deterministic encoding.

    Input:  normalized images [B, C, H, W]
    Output: [T, B, C, H, W]

    This repeats each normalized image across time instead
    of adding random binary noise.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor.")

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images shaped [B, 3, H, W].")

    if not isinstance(time_steps, int) or time_steps < 1:
        raise ValueError("time_steps must be a positive integer.")

    return images.unsqueeze(0).expand(
        time_steps, -1, -1, -1, -1
    ).contiguous()


def convert_batch_to_spikes(images, labels=None, time_steps=TIME_STEPS):
    encoded = rate_encode(images, time_steps)

    if labels is None:
        return encoded

    return encoded, labels
