
import torch
from sklearn.metrics import accuracy_score
from spikingjelly.activation_based import functional

from models.hybrid_snn_legacy import HybridSNN
from preprocessing.data_loader import (
    validation_loader,
    NUM_CLASSES,
    DEVICE,
)

CHECKPOINT_PATH = "checkpoints/best_model_weighted_snn.pth"
TIME_STEPS = 10

torch.manual_seed(42)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model = HybridSNN(
    num_classes=NUM_CLASSES,
    time_steps=checkpoint.get("time_steps", TIME_STEPS),
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

all_predictions = []
all_labels = []

print("\nEvaluating with deterministic rate encoding...")

with torch.no_grad():
    for batch_index, (images, labels) in enumerate(validation_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        mean = images.new_tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = images.new_tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        rates = (images * std + mean).clamp(0, 1)

        thresholds = (
            torch.arange(TIME_STEPS, device=DEVICE)
            .view(TIME_STEPS, 1, 1, 1, 1)
            / TIME_STEPS
        )

        spikes = (
            thresholds < rates.unsqueeze(0)
        ).float()

        functional.reset_net(model)
        outputs = model(spikes)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        functional.reset_net(model)

        if (batch_index + 1) % 50 == 0:
            print("Processed", len(all_labels), "images")

accuracy = accuracy_score(all_labels, all_predictions)

print("\n" + "=" * 45)
print("DETERMINISTIC ENCODER VALIDATION")
print("=" * 45)
print("Images evaluated:", len(all_labels))
print(f"Validation accuracy: {accuracy * 100:.2f}%")
