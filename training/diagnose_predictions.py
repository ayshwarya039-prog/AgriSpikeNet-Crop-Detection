
import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.data_loader import test_dataset, CLASS_NAMES, DEVICE
from preprocessing.snn_input import convert_batch_to_spikes, TIME_STEPS
from models.hybrid_snn import HybridSNN

torch.set_num_threads(2)

checkpoint = torch.load(
    ROOT / "checkpoints" / "best_model_clean.pth",
    map_location=DEVICE,
    weights_only=False,
)

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=checkpoint["time_steps"],
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
actual_counts = Counter()
predicted_counts = Counter()
correct = total = 0

try:
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= 10:
                break

            actual_counts.update(labels.tolist())
            rng_state = torch.random.get_rng_state()
            try:
                torch.manual_seed(10000 + batch_index)
                spikes, _ = convert_batch_to_spikes(
                    images, labels, time_steps=TIME_STEPS
                )
            finally:
                torch.random.set_rng_state(rng_state)

            functional.reset_net(model)
            outputs = model(spikes.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().tolist()
            predicted_counts.update(predictions)
            correct += sum(
                p == y for p, y in zip(predictions, labels.tolist())
            )
            total += len(labels)
            functional.reset_net(model)

finally:
    functional.reset_net(model)

print("\nImages checked:", total)
print("Correct:", correct)
print("Accuracy:", f"{100 * correct / total:.2f}%")
print("\nActual classes in these images:")
for index, count in actual_counts.most_common():
    print(f"{index}: {CLASS_NAMES[index]} ({count})")

print("\nPredicted classes:")
for index, count in predicted_counts.most_common():
    print(f"{index}: {CLASS_NAMES[index]} ({count})")
