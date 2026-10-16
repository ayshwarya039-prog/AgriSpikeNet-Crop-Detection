
import sys
import csv
import random
from pathlib import Path

# Make the project root visible to Python
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from preprocessing.data_loader import (
    train_dataset,
    validation_dataset,
    CLASS_NAMES,
    DEVICE,
)
from preprocessing.snn_input import convert_batch_to_spikes, TIME_STEPS
from models.hybrid_snn import HybridSNN


# ---------------- CONFIGURATION ----------------

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "best_model_attention.pth"
HISTORY_PATH = RESULTS_DIR / "training_history_attention.csv"

BATCH_SIZE = 4
EPOCHS = 5
MAX_TRAIN_BATCHES = 250
MAX_VAL_BATCHES = 60
LEARNING_RATE = 0.0001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------- DATA LOADERS ----------------

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


# ---------------- MODEL ----------------

model = HybridSNN(
    num_classes=len(CLASS_NAMES),
    time_steps=TIME_STEPS,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_loss = float("inf")
history = []


# ---------------- TRAINING FUNCTION ----------------

def run_epoch(loader, training, max_batches):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Convert images into spike sequences
        spikes = convert_batch_to_spikes(
            images,
            time_steps=TIME_STEPS,
        )

        if isinstance(spikes, tuple):
            spikes = spikes[0]

        optimizer.zero_grad(set_to_none=True)

        # Reset LIF neuron state for every batch
        functional.reset_net(model)

        with torch.set_grad_enabled(training):
            outputs = model(spikes)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total_images += batch_size

    functional.reset_net(model)

    if total_images == 0:
        raise RuntimeError(
            "No images were processed. Check the dataset and data loader."
        )

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy


# ---------------- MAIN TRAINING LOOP ----------------

print("=" * 55)
print("AgriSpikeNet - Attention Hybrid SNN Training")
print("=" * 55)
print(f"Device: {DEVICE}")
print(f"Number of classes: {len(CLASS_NAMES)}")
print(f"Training images available: {len(train_dataset)}")
print(f"Validation images available: {len(validation_dataset)}")
print(f"Time steps: {TIME_STEPS}")
print(f"Epochs: {EPOCHS}")
print(f"Training batches per epoch: at most {MAX_TRAIN_BATCHES}")
print(f"Validation batches per epoch: at most {MAX_VAL_BATCHES}")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_accuracy = run_epoch(
        train_loader,
        training=True,
        max_batches=MAX_TRAIN_BATCHES,
    )

    with torch.no_grad():
        val_loss, val_accuracy = run_epoch(
            validation_loader,
            training=False,
            max_batches=MAX_VAL_BATCHES,
        )

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
    }

    history.append(record)

    print(
        f"\nEpoch {epoch}/{EPOCHS}\n"
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%}\n"
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2%}"
    )

    # Save the best checkpoint without touching existing models
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": len(CLASS_NAMES),
                "class_names": list(CLASS_NAMES),
                "time_steps": TIME_STEPS,
                "epochs_completed": epoch,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "model_config": {
                    "architecture": "HybridSNN",
                    "temporal_attention": True,
                },
            },
            MODEL_PATH,
        )

        print(f"Best checkpoint saved: {MODEL_PATH.name}")


# ---------------- SAVE TRAINING HISTORY ----------------

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ],
    )

    writer.writeheader()
    writer.writerows(history)


# ---------------- COMPLETION ----------------

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)
print(f"Best model: {MODEL_PATH}")
print(f"Training history: {HISTORY_PATH}")
print(f"Best validation loss: {best_val_loss:.4f}")
print("=" * 55)
