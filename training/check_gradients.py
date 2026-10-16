import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import torch
from preprocessing.data_loader import train_loader, DEVICE
from preprocessing.snn_input import convert_batch_to_spikes, TIME_STEPS
from models.hybrid_snn import HybridSNN
from spikingjelly.activation_based import functional

model = HybridSNN(num_classes=38, time_steps=TIME_STEPS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()

images, labels = next(iter(train_loader))
images, labels = images[:4].to(DEVICE), labels[:4].to(DEVICE)

spikes, labels = convert_batch_to_spikes(
    images, labels, time_steps=TIME_STEPS
)

functional.reset_net(model)
optimizer.zero_grad(set_to_none=True)

outputs = model(spikes)
loss = criterion(outputs, labels)
loss.backward()

print("\nLoss:", round(loss.item(), 4))
print("Output shape:", tuple(outputs.shape))

for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(
            name,
            "| gradient mean:",
            round(parameter.grad.abs().mean().item(), 8),
            "| gradient exists:",
            bool(torch.isfinite(parameter.grad).all().item()),
        )
    else:
        print(name, "| gradient is None")

functional.reset_net(model)
