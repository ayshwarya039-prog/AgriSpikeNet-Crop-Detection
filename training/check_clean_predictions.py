
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from spikingjelly.activation_based import functional

from preprocessing.data_loader import (
    validation_dataset,
    CLASS_NAMES,
    DEVICE,
)
from preprocessing.snn_input import convert_batch_to_spikes
from models.hybrid_snn import HybridSNN

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model_clean.pth"


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=10,
    ).to(DEVICE)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    print("Checkpoint loaded:", CHECKPOINT.name)
    print("Number of classes:", len(CLASS_NAMES))
    print("Validation images available:", len(validation_dataset))

    # Check a small sample first
    sample_count = min(32, len(validation_dataset))
    correct = 0

    try:
        with torch.no_grad():
            for index in range(sample_count):
                image, label = validation_dataset[index]
                images = image.unsqueeze(0).to(DEVICE)
                labels = torch.tensor(
                    [int(label)], device=DEVICE
                )

                # Reproduce the encoder's expected input format
                spikes = convert_batch_to_spikes(
                    images, time_steps=10
                )

                functional.reset_net(model)
                outputs = model(spikes)
                prediction = outputs.argmax(dim=1)

                correct += int(
                    (prediction == labels).sum().item()
                )

                print(
                    f"{index + 1:02d}: "
                    f"actual={CLASS_NAMES[int(label)]}, "
                    f"predicted={CLASS_NAMES[int(prediction.item())]}"
                )

        print(
            f"\nSample accuracy: {correct}/{sample_count} "
            f"({100 * correct / max(sample_count, 1):.2f}%)"
        )
        print("This is a small diagnostic sample, not final test accuracy.")

    finally:
        functional.reset_net(model)


if __name__ == "__main__":
    main()
