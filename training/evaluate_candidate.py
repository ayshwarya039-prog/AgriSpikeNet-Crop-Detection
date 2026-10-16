
from pathlib import Path
import sys
import torch
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hybrid_snn import HybridSNN
from preprocessing.data_loader import (
    NUM_CLASSES, DEVICE, validation_dataset
)
from preprocessing.snn_input import (
    convert_batch_to_spikes, TIME_STEPS
)
from torch.utils.data import DataLoader, Subset

torch.manual_seed(42)

# Randomly select 400 validation images.
count = min(400, len(validation_dataset))
indices = torch.randperm(len(validation_dataset))[:count].tolist()
sample_loader = DataLoader(
    Subset(validation_dataset, indices),
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

checkpoint = ROOT / "checkpoints" / "best_model_weighted_snn_v2.pth"
saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)

model = HybridSNN(
    num_classes=NUM_CLASSES,
    time_steps=TIME_STEPS,
).to(DEVICE)

model.load_state_dict(saved["model_state_dict"], strict=True)
model.eval()

correct = 0
total = 0
class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)

print("Evaluating:", checkpoint.name)
print("Device:", DEVICE)
print("Validation sample:", count)

try:
    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(sample_loader, 1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            spikes, labels = convert_batch_to_spikes(
                images, labels, time_steps=TIME_STEPS
            )
            spikes = spikes.to(DEVICE)

            functional.reset_net(model)
            outputs = model(spikes)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

            for label, prediction in zip(labels, predictions):
                class_counts[label.item()] += 1
                if label.item() == prediction.item():
                    class_correct[label.item()] += 1

            if batch_number % 25 == 0:
                print(f"Processed {total}/{count} images")

    print(f"\nSample accuracy: {100 * correct / total:.2f}%")
    print("\nPer-class accuracy for classes represented in the sample:")
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            accuracy = 100 * class_correct[i] / class_counts[i]
            print(f"Class {i:2d}: {accuracy:6.2f}% "
                  f"({class_correct[i]}/{class_counts[i]})")

    print("\nEvaluation completed.")

finally:
    functional.reset_net(model)
