
from pathlib import Path
import sys
import torch
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hybrid_snn import HybridSNN
from preprocessing.data_loader import (
    NUM_CLASSES, DEVICE, validation_loader
)
from preprocessing.snn_input import (
    convert_batch_to_spikes, TIME_STEPS
)

def load_weights(path, model):
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in saved:
        saved = saved["model_state_dict"]
    elif "state_dict" in saved:
        saved = saved["state_dict"]
    model.load_state_dict(saved, strict=True)
    return model

images, labels = next(iter(validation_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)
spikes, labels = convert_batch_to_spikes(
    images, labels, time_steps=TIME_STEPS
)
spikes = spikes.to(DEVICE)

for filename in [
    "best_model.pth",
    "best_model_weighted_snn_v2.pth",
]:
    path = ROOT / "checkpoints" / filename
    model = HybridSNN(
        num_classes=NUM_CLASSES,
        time_steps=TIME_STEPS,
    ).to(DEVICE)

    print("\nCheckpoint:", filename)

    try:
        load_weights(path, model)
        model.eval()
        functional.reset_net(model)

        with torch.inference_mode():
            outputs = model(spikes)
            predictions = outputs.argmax(dim=1)

        accuracy = (predictions == labels).float().mean().item() * 100
        print(f"Same-batch accuracy: {accuracy:.2f}%")
        print("Unique predicted classes:",
              torch.unique(predictions).numel())
        print("Predicted labels:", predictions.cpu().tolist())

    except Exception as error:
        print("FAILED:", repr(error))

    finally:
        functional.reset_net(model)
