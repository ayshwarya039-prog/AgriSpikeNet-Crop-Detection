
import sys
from pathlib import Path

# --------------------------------------------------
# 1. Add the project root to Python's import path
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.data_loader import (
    train_dataset,
    validation_dataset,
    CLASS_NAMES,
    DEVICE,
)

from preprocessing.snn_input import (
    convert_batch_to_spikes,
    TIME_STEPS,
)

from models.hybrid_snn import HybridSNN
from spikingjelly.activation_based import functional


# --------------------------------------------------
# 2. Configuration for a CPU-friendly first run
# --------------------------------------------------
BATCH_SIZE = 2
EPOCHS = 5
MAX_TRAIN_BATCHES = 20
MAX_VAL_BATCHES = 5
LEARNING_RATE = 0.0005

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("AgriSpikeNet - Hybrid SNN Training")
print("=" * 60)
print("Project root :", PROJECT_ROOT)
print("Device       :", DEVICE)
print("Classes      :", len(CLASS_NAMES))
print("Time steps   :", TIME_STEPS)
print("Epochs       :", EPOCHS)
print("Batch size   :", BATCH_SIZE)
print("Train batches per epoch:", MAX_TRAIN_BATCHES)
print("Validation batches per epoch:", MAX_VAL_BATCHES)


# --------------------------------------------------
# 3. Create data loaders
# --------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

if len(train_dataset) == 0:
    raise RuntimeError("Training dataset is empty. Check preprocessing/data_loader.py.")

if len(validation_dataset) == 0:
    raise RuntimeError("Validation dataset is empty. Check preprocessing/data_loader.py.")


# --------------------------------------------------
# 4. Initialize the Hybrid SNN model
# --------------------------------------------------
model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=TIME_STEPS,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_validation_accuracy = 0.0


# --------------------------------------------------
# 5. Convert images into spikes
# --------------------------------------------------
def prepare_spikes(images, labels):
    """
    Converts a batch of images into spike sequences.
    Expected spike shape: [time_steps, batch, channels, height, width]
    """
    result = convert_batch_to_spikes(
        images,
        labels,
        time_steps=TIME_STEPS,
    )

    # The encoder may return spikes alone or (spikes, labels).
    if isinstance(result, (tuple, list)):
        spikes = result[0]
    else:
        spikes = result

    if not isinstance(spikes, torch.Tensor):
        raise TypeError("The spike encoder did not return a PyTorch tensor.")

    return spikes.to(DEVICE)


# --------------------------------------------------
# 6. Train for one epoch
# --------------------------------------------------
def train_one_epoch():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    processed_batches = 0

    for batch_index, (images, labels) in enumerate(train_loader):
        if batch_index >= MAX_TRAIN_BATCHES:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad(set_to_none=True)

        spikes = prepare_spikes(images, labels)

        # Reset the internal state of the spiking neurons.
        functional.reset_net(model)

        outputs = model(spikes)

        if outputs.ndim != 2:
            raise RuntimeError(
                f"Expected model output [batch, classes], "
                f"but received shape {tuple(outputs.shape)}."
            )

        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training loss became NaN or infinity. "
                "Stop and inspect the model and input data."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        processed_batches += 1

        if batch_index == 0:
            print(
                "First batch shapes:",
                "images =", tuple(images.shape),
                "| spikes =", tuple(spikes.shape),
                "| outputs =", tuple(outputs.shape),
            )

    if total == 0:
        raise RuntimeError("No training batches were processed.")

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy, processed_batches


# --------------------------------------------------
# 7. Validate the model
# --------------------------------------------------
def validate_one_epoch():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    processed_batches = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(validation_loader):
            if batch_index >= MAX_VAL_BATCHES:
                break

            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            spikes = prepare_spikes(images, labels)

            functional.reset_net(model)

            outputs = model(spikes)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size
            processed_batches += 1

    if total == 0:
        raise RuntimeError("No validation batches were processed.")

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy, processed_batches


# --------------------------------------------------
# 8. Training loop and checkpoint saving
# --------------------------------------------------
history_rows = []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 40)

    train_loss, train_accuracy, train_batches = train_one_epoch()

    val_loss, val_accuracy, val_batches = validate_one_epoch()

    print(
        f"Train | loss: {train_loss:.4f} "
        f"| accuracy: {train_accuracy * 100:.2f}% "
        f"| batches: {train_batches}"
    )

    print(
        f"Valid | loss: {val_loss:.4f} "
        f"| accuracy: {val_accuracy * 100:.2f}% "
        f"| batches: {val_batches}"
    )

    history_rows.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
    })

    # Save the best model based on validation accuracy.
    if val_accuracy > best_validation_accuracy:
        best_validation_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_accuracy": best_validation_accuracy,
                "class_names": list(CLASS_NAMES),
                "num_classes": len(CLASS_NAMES),
                "time_steps": TIME_STEPS,
            },
            BEST_MODEL_PATH,
        )

        print(f"Best model saved to: {BEST_MODEL_PATH}")

    # Save training history after every epoch.
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


# --------------------------------------------------
# 9. Training summary
# --------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print(f"Best validation accuracy: {best_validation_accuracy * 100:.2f}%")
print(f"Model checkpoint: {BEST_MODEL_PATH}")
print(f"Training history: {HISTORY_PATH}")
print(
    "\nNote: This is a limited-batch run for checking the pipeline. "
    "Its accuracy is not the final full-dataset result."
)