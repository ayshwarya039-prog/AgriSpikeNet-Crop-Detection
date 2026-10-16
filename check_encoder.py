
import torch

from preprocessing.data_loader import (
    validation_loader,
    DEVICE,
)
from preprocessing.snn_input import rate_encode

images, labels = next(iter(validation_loader))
images = images.to(DEVICE)

mean = images.new_tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

std = images.new_tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)

rates = ((images * std + mean).clamp(0, 1))

# Deterministic rate encoding for comparison
time_steps = 10
deterministic_spikes = (
    torch.arange(time_steps, device=DEVICE)
    .view(time_steps, 1, 1, 1, 1)
    < (rates.unsqueeze(0) * time_steps)
).float()

random_spikes = rate_encode(images, time_steps)

print("Images:", tuple(images.shape))
print("Deterministic spikes:", tuple(deterministic_spikes.shape))
print("Random spikes:", tuple(random_spikes.shape))
print(
    "Deterministic mean spike rate:",
    deterministic_spikes.mean().item(),
)
print(
    "Random mean spike rate:",
    random_spikes.mean().item(),
)
print(
    "Are the two encodings identical?",
    torch.equal(deterministic_spikes, random_spikes),
)
