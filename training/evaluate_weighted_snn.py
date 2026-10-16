
import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.data_loader import (
    validation_dataset,
    CLASS_NAMES,
    DEVICE,
)

from preprocessing.snn_input import (
    convert_batch_to_spikes,
    TIME_STEPS,
)

from models.hybrid_snn import HybridSNN

torch.set_num_threads(2)

CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "best_model_weighted_snn.pth"
)

# Evaluate 1,000 shuffled validation images.
SAMPLE_SIZE = 1000
BATCH_SIZE = 4

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=checkpoint["time_steps"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

generator = torch.Generator().manual_seed(2026)

loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
    num_workers=0,
)

actual_counts = Counter()
predicted_counts = Counter()
correct_by_class = Counter()
total_by_class = Counter()

correct = 0
total = 0

try:
    with torch.no_grad():
        for images, labels in loader:
            if total >= SAMPLE_SIZE:
                break

            remaining = SAMPLE_SIZE - total
            images = images[:remaining].to(DEVICE)
            labels = labels[:remaining].to(DEVICE)

            spikes, labels = convert_batch_to_spikes(
                images,
                labels,
                time_steps=TIME_STEPS,
            )

            functional.reset_net(model)

            outputs = model(spikes.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().tolist()
            actual = labels.cpu().tolist()

            for prediction, target in zip(predictions, actual):
                actual_counts[target] += 1
                predicted_counts[prediction] += 1
                total_by_class[target] += 1

                if prediction == target:
                    correct += 1
                    correct_by_class[target] += 1

            total += len(actual)
            functional.reset_net(model)

finally:
    functional.reset_net(model)

print("\n========== WEIGHTED HYBRID SNN EVALUATION ==========")
print("Checkpoint epoch:", checkpoint["epochs_completed"])
print("Images evaluated:", total)
print("Correct predictions:", correct)
print(f"Validation accuracy: {100 * correct / total:.2f}%")

print("\n========== PER-CLASS RECALL ==========")

recalls = []

for index, count in sorted(total_by_class.items()):
    recall = correct_by_class[index] / count
    recalls.append(recall)

    print(
        f"{CLASS_NAMES[index]}: "
        f"{correct_by_class[index]}/{count} "
        f"({100 * recall:.2f}%)"
    )

print("\nClasses represented:", len(actual_counts))
print("Classes predicted:", len(predicted_counts))

print("\n========== MOST COMMON PREDICTIONS ==========")

for index, count in predicted_counts.most_common(10):
    print(f"{CLASS_NAMES[index]}: {count}")

if recalls:
    print(
        "\nMacro-average recall across represented classes:",
        f"{100 * sum(recalls) / len(recalls):.2f}%",
    )
