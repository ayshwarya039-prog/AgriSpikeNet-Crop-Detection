import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from preprocessing.data_loader import validation_dataset, CLASS_NAMES, DEVICE
from preprocessing.snn_input_backup import convert_batch_to_spikes
from models.hybrid_snn_legacy import HybridSNN

torch.set_num_threads(2)

checkpoint_path = ROOT / "checkpoints" / "best_model_weighted_snn_v2.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=checkpoint["time_steps"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

loader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

correct = 0
total = 0
max_images = 1000

try:
    with torch.no_grad():
        for images, labels in loader:
            if total >= max_images:
                break

            count = min(images.size(0), max_images - total)
            images = images[:count].to(DEVICE)
            labels = labels[:count].to(DEVICE)

            spikes = convert_batch_to_spikes(
                images,
                time_steps=checkpoint["time_steps"],
            )

            functional.reset_net(model)
            outputs = model(spikes.to(DEVICE))
            correct += (outputs.argmax(1) == labels).sum().item()
            total += count
            functional.reset_net(model)
finally:
    functional.reset_net(model)

print("\nCheckpoint:", checkpoint_path.name)
print("Images evaluated:", total)
print("Saved accuracy:", checkpoint.get("validation_accuracy", "unknown"))
print(f"Re-evaluated accuracy: {100 * correct / total:.2f}%")
