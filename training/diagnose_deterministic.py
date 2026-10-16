
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

checkpoint_path = (
    ROOT / "checkpoints" / "best_model_deterministic.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False,
)

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=checkpoint["time_steps"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Shuffle deterministically to sample multiple classes.
generator = torch.Generator().manual_seed(42)

loader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=True,
    generator=generator,
    num_workers=0,
)

actual_counts = Counter()
predicted_counts = Counter()
correct = 0
total = 0

try:
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= 100:
                break

            spikes, labels = convert_batch_to_spikes(
                images,
                labels,
                time_steps=TIME_STEPS,
            )

            functional.reset_net(model)

            outputs = model(spikes.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().tolist()
            actual = labels.tolist()

            actual_counts.update(actual)
            predicted_counts.update(predictions)

            correct += sum(
                p == y for p, y in zip(predictions, actual)
            )
            total += len(actual)

            functional.reset_net(model)

finally:
    functional.reset_net(model)

print("\nCheckpoint epoch:", checkpoint["epochs_completed"])
print("Images evaluated:", total)
print("Correct predictions:", correct)
print("Validation accuracy:", f"{100 * correct / total:.2f}%")

print("\nMost common actual classes:")
for index, count in actual_counts.most_common(10):
    print(f"{CLASS_NAMES[index]}: {count}")

print("\nPredicted classes:")
for index, count in predicted_counts.most_common(15):
    print(f"{CLASS_NAMES[index]}: {count}")
