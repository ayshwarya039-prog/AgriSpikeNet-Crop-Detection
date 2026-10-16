import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from preprocessing.data_loader import (
    validation_dataset,
    CLASS_NAMES,
    DEVICE,
)
from preprocessing.snn_input import convert_batch_to_spikes
from models.hybrid_snn_legacy import HybridSNN


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "best_model_weighted_snn_v2.pth"
)
RESULTS_DIR = ROOT / "results"

BATCH_SIZE = 4
SEED = 42


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("Loading validation dataset...")
    print("Validation images:", len(validation_dataset))
    print("Number of classes:", len(CLASS_NAMES))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    time_steps = int(checkpoint.get("time_steps", 10))

    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=time_steps,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "full_validation_predictions.csv"
    report_path = RESULTS_DIR / "full_validation_report.txt"

    correct = 0
    total = 0
    class_correct = [0] * len(CLASS_NAMES)
    class_total = [0] * len(CLASS_NAMES)
    names = list(CLASS_NAMES)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_index",
            "true_label",
            "predicted_label",
            "correct",
        ])

        with torch.inference_mode():
            for batch_number, (images, labels) in enumerate(
                loader, start=1
            ):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                spikes = convert_batch_to_spikes(
                    images,
                    time_steps=time_steps,
                )

                outputs = model(spikes)

                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                predictions = outputs.argmax(dim=1)
                matches = predictions.eq(labels)

                batch_size = labels.size(0)
                start_index = total
                correct += int(matches.sum().item())
                total += batch_size

                true_ids = labels.cpu().tolist()
                predicted_ids = predictions.cpu().tolist()
                match_values = matches.cpu().tolist()

                for offset, (true_id, pred_id, matched) in enumerate(
                    zip(true_ids, predicted_ids, match_values)
                ):
                    class_total[true_id] += 1

                    if matched:
                        class_correct[true_id] += 1

                    writer.writerow([
                        start_index + offset + 1,
                        names[true_id],
                        names[pred_id],
                        matched,
                    ])

                if batch_number % 100 == 0 or total == len(
                    validation_dataset
                ):
                    print(
                        f"Progress: {total}/{len(validation_dataset)} "
                        f"| Accuracy: {100 * correct / total:.2f}%"
                    )

    accuracy = correct / total if total else 0.0

    report = [
        "AgriSpikeNet Full Validation Report",
        "===================================",
        f"Checkpoint: {CHECKPOINT_PATH.name}",
        f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Checkpoint recorded accuracy: "
        f"{checkpoint.get('validation_accuracy', 'unknown')}",
        f"Encoder: preprocessing.snn_input",
        f"Random seed: {SEED}",
        f"Time steps: {time_steps}",
        f"Images evaluated: {total}",
        f"Correct predictions: {correct}",
        f"Incorrect predictions: {total - correct}",
        f"Full validation accuracy: {accuracy * 100:.2f}%",
        "",
        "Per-class accuracy",
        "------------------",
    ]

    for index, name in enumerate(names):
        count = class_total[index]
        class_accuracy = (
            class_correct[index] / count if count else 0.0
        )
        report.append(
            f"{name}: {class_correct[index]}/{count} "
            f"({class_accuracy * 100:.2f}%)"
        )

    report.extend([
        "",
        "Note: This evaluation uses the current encoder code.",
        "Confirm it matches the original training encoder before",
        "reporting the result as final scientific accuracy.",
    ])

    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report))
    print("\nReport saved to:", report_path)
    print("Predictions saved to:", csv_path)


if __name__ == "__main__":
    main()
