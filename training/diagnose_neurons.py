
import sys
from pathlib import Path

import torch
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.data_loader import train_dataset, CLASS_NAMES, DEVICE
from preprocessing.snn_input import convert_batch_to_spikes, TIME_STEPS
from models.hybrid_snn import HybridSNN

torch.manual_seed(42)

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=TIME_STEPS,
).to(DEVICE)

# Use the existing checkpoint without modifying it.
path = ROOT / "checkpoints" / "best_model.pth"
checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

images = torch.stack([
    train_dataset[i][0] for i in range(4)
]).to(DEVICE)

labels = torch.tensor([
    train_dataset[i][1] for i in range(4)
], dtype=torch.long, device=DEVICE)

spikes = convert_batch_to_spikes(
    images, labels, time_steps=TIME_STEPS
)[0]

functional.reset_net(model)

with torch.no_grad():
    outputs = model(spikes)

print("\n=== Neuron Diagnostic Results ===")
print("Input spike rate:", round(spikes.mean().item(), 4))
print("Output shape:", tuple(outputs.shape))
print("Output finite:", bool(torch.isfinite(outputs).all()))
print("Output scores:", outputs.cpu().tolist())
print("Predicted classes:", outputs.argmax(dim=1).cpu().tolist())
print("Actual classes:", labels.cpu().tolist())
print("LIF spike rates:", model.last_spike_rates)
print("Overall LIF spike rate:", model.last_overall_spike_rate)

functional.reset_net(model)
