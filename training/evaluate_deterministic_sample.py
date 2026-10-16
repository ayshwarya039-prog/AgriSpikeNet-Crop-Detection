import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from spikingjelly.activation_based import functional
from preprocessing import data_loader
from preprocessing.snn_input_backup import convert_batch_to_spikes
from models.hybrid_snn_legacy import HybridSNN

device = torch.device("cpu")
checkpoint_path = "checkpoints/best_model_weighted_snn_v2.pth"

checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=False
)
state_dict = checkpoint.get("model_state_dict", checkpoint)

class_names = data_loader.CLASS_NAMES
model = HybridSNN(
    num_classes=len(class_names),
    time_steps=checkpoint.get("time_steps", 10),
).to(device)

model.load_state_dict(state_dict, strict=True)
model.eval()

correct = 0
total = 0
max_images = 400

with torch.no_grad():
    for images, labels in data_loader.validation_loader:
        remaining = max_images - total
        if remaining <= 0:
            break

        images = images[:remaining].to(device)
        labels = labels[:remaining].to(device)

        spikes = convert_batch_to_spikes(
            images, time_steps=model.time_steps
        )

        functional.reset_net(model)
        try:
            outputs = model(spikes)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        finally:
            functional.reset_net(model)

print("\nCheckpoint:", checkpoint_path)
print("Encoder: deterministic rate (backup)")
print("Validation images tested:", total)
print(f"Validation accuracy: {100 * correct / total:.2f}%")
