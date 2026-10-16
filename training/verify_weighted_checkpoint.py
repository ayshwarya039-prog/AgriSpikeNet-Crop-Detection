
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import (
    validation_dataset,
    CLASS_NAMES,
    DEVICE,
)
from preprocessing.snn_input_backup import convert_batch_to_spikes
from models.hybrid_snn_legacy import HybridSNN

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model_weighted_snn.pth"

def main():
    torch.set_num_threads(2)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    print("Checkpoint:", CHECKPOINT.name)
    print("Saved accuracy:", checkpoint.get("validation_accuracy"))
    print("Saved encoder:", checkpoint.get("model_config", {}).get("encoder"))

    saved_classes = checkpoint.get("class_names", [])
    if list(saved_classes) != list(CLASS_NAMES):
        raise RuntimeError(
            "Checkpoint class order differs from the current dataset."
        )

    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=checkpoint.get("time_steps", 10),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loader):
                if batch_index >= 25:
                    break

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                spikes = convert_batch_to_spikes(
                    images,
                    time_steps=checkpoint.get("time_steps", 10),
                )

                functional.reset_net(model)
                outputs = model(spikes.to(DEVICE))
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                print("True labels:", labels.tolist())
                print("Predictions:", predictions.tolist())
                print("Output range:", outputs.min().item(), outputs.max().item())

        functional.reset_net(model)

        print("Validation images tested:", total)
        print("Correct predictions:", correct)
        print(
            "Accuracy on this sample: "
            f"{100 * correct / total:.2f}%"
        )
        print("Note: this is a 100-image diagnostic sample, not final accuracy.")

    finally:
        functional.reset_net(model)

if __name__ == "__main__":
    main()

