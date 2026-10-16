
import sys
from pathlib import Path

import torch
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.data_loader import (
    validation_dataset,
    CLASS_NAMES,
    DEVICE,
)
from preprocessing.snn_input import (
    convert_batch_to_spikes,
    TIME_STEPS,
)
from models.hybrid_snn import HybridSNN

print("\n=== AgriSpikeNet Model Check ===")
print("Device:", DEVICE)
print("Classes:", len(CLASS_NAMES))
print("Validation images:", len(validation_dataset))

checkpoint_path = ROOT / "checkpoints" / "best_model.pth"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False,
)

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=TIME_STEPS,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint loaded successfully.")
print("Checkpoint classes:", checkpoint.get("num_classes"))
print("Checkpoint validation loss:",
      checkpoint.get("validation_loss"))

correct = 0
total = 0

# Test only 4 batches to keep this quick.
for index in range(min(4, (len(validation_dataset) + 7) // 8)):
    start = index * 8
    end = min(start + 8, len(validation_dataset))

    images = torch.stack([
        validation_dataset[i][0]
        for i in range(start, end)
    ]).to(DEVICE)

    labels = torch.tensor([
        validation_dataset[i][1]
        for i in range(start, end)
    ], dtype=torch.long, device=DEVICE)

    spikes = convert_batch_to_spikes(
        images,
        time_steps=TIME_STEPS,
    )

    functional.reset_net(model)

    with torch.no_grad():
        outputs = model(spikes)
        predictions = outputs.argmax(dim=1)

    correct += (predictions == labels).sum().item()
    total += labels.size(0)

    print(
        f"Batch {index + 1}: "
        f"{(predictions == labels).sum().item()}/{labels.size(0)} correct"
    )

    functional.reset_net(model)

print("\n=== Results ===")
print("Correct predictions:", correct, "out of", total)
print("Validation accuracy:", round(100 * correct / total, 2), "%")
print("This is only a small diagnostic sample.")
