
import torch

TIME_STEPS = 10


def rate_encode(images, time_steps=TIME_STEPS):
    """Convert normalized RGB images into binary spike trains."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor")

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images with shape [B, 3, H, W]")

    if time_steps < 1:
        raise ValueError("time_steps must be at least 1")

    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    rates = (images * std + mean).clamp(0, 1)
    random_values = torch.rand(
        (time_steps, *rates.shape),
        device=images.device
    )
    spikes = (random_values < rates.unsqueeze(0)).to(torch.float32)

    return spikes


def convert_batch_to_spikes(
    images, labels=None, time_steps=TIME_STEPS
):
    spikes = rate_encode(images, time_steps)

    if labels is None:
        return spikes

    return spikes, labels
