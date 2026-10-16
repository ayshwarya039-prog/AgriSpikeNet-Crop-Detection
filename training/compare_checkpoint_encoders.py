
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.data_loader import validation_dataset, CLASS_NAMES, DEVICE
from preprocessing.snn_input import rate_encode as binary_encoder
from preprocessing.snn_input_backup import rate_encode as deterministic_encoder
from models.hybrid_snn_legacy import HybridSNN

CHECKPOINT = ROOT / "checkpoints" / "best_model_weighted_snn.pth"
SAMPLE_SIZE = 100
BATCH_SIZE = 4


def evaluate(model, loader, encoder, time_steps):
    correct = 0
    total = 0

    model.eval()

    try:
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                spikes = encoder(images, time_steps=time_steps)

                functional.reset_net(model)
                outputs = model(spikes.to(DEVICE))
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                functional.reset_net(model)

    finally:
        functional.reset_net(model)

    return correct, total


def main():
    torch.set_num_threads(2)

    saved = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if list(saved["class_names"]) != list(CLASS_NAMES):
        raise RuntimeError(
            "Checkpoint class names/order do not match the dataset."
        )

    time_steps = saved.get("time_steps", 10)

    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=time_steps,
    ).to(DEVICE)

    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    # Fixed, ordered sample so both encoders see the same images.
    sample = torch.utils.data.Subset(
        validation_dataset,
        range(min(SAMPLE_SIZE, len(validation_dataset))),
    )

    loader = DataLoader(
        sample,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\nCheckpoint:", CHECKPOINT.name)
    print("Recorded validation accuracy:",
          f"{saved.get('validation_accuracy', 0) * 100:.2f}%")
    print("Images in this diagnostic:", len(sample))

    for name, encoder in [
        ("Random binary encoder", binary_encoder),
        ("Deterministic rate encoder", deterministic_encoder),
    ]:
        correct, total = evaluate(
            model, loader, encoder, time_steps
        )
        accuracy = 100 * correct / total if total else 0

        print(f"\n{name}")
        print(f"Correct: {correct}/{total}")
        print(f"Diagnostic accuracy: {accuracy:.2f}%")

    print("\nThis is a diagnostic comparison, not final model accuracy.")


if __name__ == "__main__":
    main()

