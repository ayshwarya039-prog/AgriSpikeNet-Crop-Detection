
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import test_dataset, CLASS_NAMES, DEVICE
from preprocessing.snn_input import convert_batch_to_spikes, TIME_STEPS
from models.hybrid_snn import HybridSNN

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model_clean.pth"

torch.set_num_threads(2)
loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=True,
    generator=torch.Generator().manual_seed(42)
)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=checkpoint["time_steps"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = 0
total = 0
max_batches = 100

try:
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= max_batches:
                break

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Use fixed randomness for repeatable test encoding.
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
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            functional.reset_net(model)

            if (batch_index + 1) % 25 == 0:
                print(f"Test batches: {batch_index + 1}/{max_batches}")

finally:
    functional.reset_net(model)

print("\nTest evaluation finished.")
print("Images tested:", total)
print("Correct predictions:", correct)
print(f"Test accuracy: {100 * correct / total:.2f}%")
print("Note: This is a 400-image test subset, not the full test set.")
