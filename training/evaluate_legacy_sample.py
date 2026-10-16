import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from spikingjelly.activation_based import functional
from preprocessing import data_loader
from preprocessing.snn_input import convert_batch_to_spikes
from models.hybrid_snn_legacy import HybridSNN

device = torch.device("cpu")
checkpoint_path = "checkpoints/best_model_weighted_snn_v2.pth"

# The data loader creates these variables when imported.
val_loader = data_loader.validation_loader
class_names = data_loader.CLASS_NAMES

checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=False
)
state_dict = checkpoint.get("model_state_dict", checkpoint)

model = HybridSNN(
    num_classes=len(class_names), time_steps=10
).to(device)
model.load_state_dict(state_dict, strict=True)
model.eval()

correct = 0
total = 0
max_images = 400

with torch.no_grad():
    for images, labels in val_loader:
        remaining = max_images - total
        if remaining <= 0:
            break

        images = images[:remaining].to(device)
        labels = labels[:remaining].to(device)

        spikes = convert_batch_to_spikes(images, time_steps=10)
        functional.reset_net(model)
        outputs = model(spikes)

        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

print(f"\nCheckpoint: {checkpoint_path}")
print(f"Validation images tested: {total}")
print(f"Validation accuracy: {100 * correct / total:.2f}%")
