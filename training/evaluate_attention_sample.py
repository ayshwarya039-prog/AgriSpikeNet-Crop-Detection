
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hybrid_snn import HybridSNN
from preprocessing.data_loader import NUM_CLASSES, DEVICE, validation_dataset
from preprocessing.snn_input import convert_batch_to_spikes, TIME_STEPS

torch.manual_seed(42)

count = min(400, len(validation_dataset))
indices = torch.randperm(len(validation_dataset))[:count].tolist()
loader = DataLoader(
    Subset(validation_dataset, indices),
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

checkpoint = ROOT / "checkpoints" / "best_model_attention.pth"
saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)

model = HybridSNN(num_classes=NUM_CLASSES, time_steps=TIME_STEPS).to(DEVICE)
state = saved.get("model_state_dict", saved)
model.load_state_dict(state, strict=True)
model.eval()

correct = 0
total = 0
print("Checkpoint:", checkpoint.name)
print("Evaluating", count, "validation images on", DEVICE)

try:
    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            spikes, labels = convert_batch_to_spikes(
                images, labels, time_steps=TIME_STEPS
            )
            functional.reset_net(model)
            predictions = model(spikes.to(DEVICE)).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

            if batch_number % 25 == 0:
                print(f"Processed {total}/{count}")

    print(f"Validation sample accuracy: {100 * correct / total:.2f}%")
finally:
    functional.reset_net(model)
