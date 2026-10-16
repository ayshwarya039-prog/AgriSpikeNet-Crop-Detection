
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from preprocessing.data_loader import train_loader, DEVICE
from preprocessing.snn_input import convert_batch_to_spikes, TIME_STEPS
from models.hybrid_snn import HybridSNN
from spikingjelly.activation_based import functional

torch.manual_seed(42)
torch.set_num_threads(2)

images, labels = next(iter(train_loader))
images = images[:4].to(DEVICE)
labels = labels[:4].to(DEVICE)

model = HybridSNN(num_classes=38, time_steps=TIME_STEPS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

print("Actual labels:", labels.tolist())
print("Testing whether the model can learn 4 fixed images.")

try:
    for step in range(1, 21):
        # Reuse the same binary encoding so input randomness
        # does not change between updates.
        torch.manual_seed(123)
        spikes = convert_batch_to_spikes(
            images, time_steps=TIME_STEPS
        ).to(DEVICE)

        model.train()
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(spikes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.detach().argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item() * 100

        if step == 1 or step % 5 == 0:
            print(
                f"Step {step:02d}: "
                f"loss={loss.item():.4f}, "
                f"accuracy={accuracy:.1f}%, "
                f"predictions={predictions.tolist()}"
            )

        functional.reset_net(model)

    print("Overfit diagnostic complete. No checkpoint saved.")

finally:
    functional.reset_net(model)
