
import torch
from sklearn.metrics import classification_report, accuracy_score
from spikingjelly.activation_based import functional

from models.hybrid_snn_legacy import HybridSNN
from preprocessing.data_loader import (
    validation_loader,
    NUM_CLASSES,
    DEVICE,
)
from preprocessing.snn_input import convert_batch_to_spikes


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHECKPOINT_PATH = (
    "checkpoints/best_model_weighted_snn.pth"
)
TIME_STEPS = 10
RANDOM_SEED = 42

# Fix random seeds for repeatable evaluation
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("\nLoading checkpoint...")


# --------------------------------------------------
# 2. Load the saved model
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model = HybridSNN(
    num_classes=NUM_CLASSES,
    time_steps=checkpoint.get("time_steps", TIME_STEPS),
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

print("Checkpoint loaded successfully.")
print("Number of classes:", NUM_CLASSES)
print("Time steps:", model.time_steps)
print("Device:", DEVICE)


# --------------------------------------------------
# 3. Evaluate the validation dataset
# --------------------------------------------------

all_predictions = []
all_labels = []

print("\nEvaluating validation dataset...")

with torch.no_grad():
    for batch_index, (images, labels) in enumerate(
        validation_loader
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Convert normalized images to binary spike trains
        spikes = convert_batch_to_spikes(images)

        # Support encoders that optionally return labels
        if isinstance(spikes, tuple):
            spikes = spikes[0]

        spikes = spikes.to(DEVICE)

        # Reset the stateful LIF neurons for each batch
        functional.reset_net(model)

        outputs = model(spikes)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_labels.extend(labels.cpu().tolist())

        functional.reset_net(model)

        if (batch_index + 1) % 50 == 0:
            print(
                f"Processed {len(all_labels)} "
                f"/ {len(validation_loader.dataset)} images"
            )


# --------------------------------------------------
# 4. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

correct = sum(
    predicted == actual
    for predicted, actual in zip(
        all_predictions, all_labels
    )
)

print("\n" + "=" * 55)
print("       AGRISPIKENET VALIDATION RESULTS")
print("=" * 55)

print(f"Total images       : {len(all_labels)}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions  : {len(all_labels) - correct}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 5. Print classification report
# --------------------------------------------------

class_names = [
    validation_loader.dataset.classes[i]
    for i in range(NUM_CLASSES)
]

print("\nCLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0,
    digits=4,
)

print(report)


# --------------------------------------------------
# 6. Save results without overwriting checkpoints
# --------------------------------------------------

import os

os.makedirs("results", exist_ok=True)

report_path = (
    "results/legacy_validation_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("AgriSpikeNet Validation Results\n")
    file.write("=" * 55 + "\n")
    file.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    file.write(f"Images evaluated: {len(all_labels)}\n")
    file.write(f"Correct predictions: {correct}\n")
    file.write(
        f"Validation accuracy: {accuracy * 100:.2f}%\n\n"
    )
    file.write(report)

print("\nReport saved to:", report_path)
print("Evaluation completed successfully.")
