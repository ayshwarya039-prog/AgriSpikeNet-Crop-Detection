
from pathlib import Path
import sys

import torch
from spikingjelly.activation_based import functional

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.hybrid_snn import HybridSNN
from preprocessing.data_loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    DEVICE,
    validation_loader,
)
from preprocessing.snn_input import (
    convert_batch_to_spikes,
    TIME_STEPS,
)

CHECKPOINT = PROJECT_DIR / "checkpoints" / "best_model.pth"

print("=" * 55)
print("AgriSpikeNet Checkpoint Verification")
print("=" * 55)
print("Device:", DEVICE)
print("Number of classes:", NUM_CLASSES)
print("Checkpoint:", CHECKPOINT)

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

model = HybridSNN(
    num_classes=NUM_CLASSES,
    time_steps=TIME_STEPS,
).to(DEVICE)

saved = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True,
)

# Support common checkpoint formats.
if isinstance(saved, dict) and "model_state_dict" in saved:
    state_dict = saved["model_state_dict"]
elif isinstance(saved, dict) and "state_dict" in saved:
    state_dict = saved["state_dict"]
else:
    state_dict = saved

model.load_state_dict(state_dict, strict=True)
model.eval()

images, labels = next(iter(validation_loader))
images = images.to(DEVICE)
labels = labels.to(DEVICE)

spikes, labels = convert_batch_to_spikes(
    images, labels, time_steps=TIME_STEPS
)
spikes = spikes.to(DEVICE)

functional.reset_net(model)

try:
    with torch.inference_mode():
        outputs = model(spikes)
        predictions = outputs.argmax(dim=1)

    print("\nCHECKPOINT LOADED SUCCESSFULLY")
    print("Input shape:", tuple(spikes.shape))
    print("Output shape:", tuple(outputs.shape))
    print("Predictions:", predictions.cpu().tolist())
    print("Actual labels:", labels.cpu().tolist())
    print("Batch accuracy:",
          f"{(predictions == labels).float().mean().item() * 100:.2f}%")
    print("\nCheckpoint compatibility test: PASSED")

finally:
    functional.reset_net(model)
