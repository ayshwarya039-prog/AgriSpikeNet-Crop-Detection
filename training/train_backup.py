
import sys
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

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
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 4
EPOCHS = 5

# CPU-friendly experimental limits
MAX_TRAIN_BATCHES = 500
MAX_VAL_BATCHES = 100

LEARNING_RATE = 0.0001
RANDOM_SEED = 42

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Separate filename to protect existing checkpoints
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model_deterministic.pth"
HISTORY_PATH = RESULTS_DIR / "training_history_deterministic.csv"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.set_num_threads(2)


# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------

def make_loader(dataset, shuffle):
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator if shuffle else None,
    )


train_loader = make_loader(
    train_dataset,
    shuffle=True,
)

# Shuffle validation images so the limited sample
# includes examples from different classes.
validation_loader = make_loader(
    validation_dataset,
    shuffle=True,
)


# --------------------------------------------------
# INPUT ENCODING
# --------------------------------------------------

def encode_images(images, labels, training, batch_index):
    """
    Use the same deterministic encoder during training
    and validation.

    The current encoder repeats continuous-valued image
    rates over the simulation time steps. These are not
    binary spikes.
    """
    encoded, labels = convert_batch_to_spikes(
        images,
        labels,
        time_steps=TIME_STEPS,
    )

    return encoded.to(DEVICE), labels.to(DEVICE)


# --------------------------------------------------
# RUN ONE EPOCH
# --------------------------------------------------

def run_epoch(model, loader, optimizer, training, max_batches):
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0
    batches_done = 0

    for batch_index, (images, labels) in enumerate(loader):

        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        encoded, labels = encode_images(
            images,
            labels,
            training=training,
            batch_index=batch_index,
        )

        functional.reset_net(model)

        if training:
            optimizer.zero_grad(set_to_none=True)

        try:
            with torch.set_grad_enabled(training):

                outputs = model(encoded)

                loss = nn.functional.cross_entropy(
                    outputs,
                    labels,
                )

                if training:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size
            batches_done += 1

            if (batch_index + 1) % 50 == 0:
                phase = (
                    "Training"
                    if training
                    else "Validation"
                )

                print(
                    f"{phase} batch {batch_index + 1}: "
                    f"loss={loss.item():.4f}"
                )

        finally:
            functional.reset_net(model)

        del images, labels, encoded, outputs, loss

    if total == 0:
        raise RuntimeError(
            "No batches were processed. Check your dataset."
        )

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy, batches_done


# --------------------------------------------------
# SAVE BEST CHECKPOINT
# --------------------------------------------------

def save_checkpoint(model, epoch, val_loss, val_accuracy):

    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")

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
                "input_size": 224,
                "internal_frame_size": 64,
                "interpolation": "bilinear",
                "encoder": "deterministic_rate",
            },
        },
        temporary_path,
    )

    temporary_path.replace(CHECKPOINT_PATH)


# --------------------------------------------------
# SAVE TRAINING HISTORY
# --------------------------------------------------

def save_history(history):

    temporary_path = HISTORY_PATH.with_suffix(".tmp")

    with open(
        temporary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "train_batches",
            "validation_batches",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

    temporary_path.replace(HISTORY_PATH)


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AgriSpikeNet - Deterministic Encoder Experiment")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Number of classes:", len(CLASS_NAMES))
    print("Batch size:", BATCH_SIZE)
    print("Time steps:", TIME_STEPS)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Maximum training batches:", MAX_TRAIN_BATCHES)
    print("Maximum validation batches:", MAX_VAL_BATCHES)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)

    # Start from fresh weights for this separate experiment.
    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=TIME_STEPS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):

        print(f"\n{'=' * 20} EPOCH {epoch}/{EPOCHS} {'=' * 20}")

        train_loss, train_accuracy, train_count = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            training=True,
            max_batches=MAX_TRAIN_BATCHES,
        )

        val_loss, val_accuracy, val_count = run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=optimizer,
            training=False,
            max_batches=MAX_VAL_BATCHES,
        )

        print("\nEpoch results")
        print("-" * 40)
        print("Training batches:", train_count)
        print("Validation batches:", val_count)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "train_batches": train_count,
                "validation_batches": val_count,
            }
        )

        # Save the model with the lowest validation loss.
        if val_loss < best_val_loss:

            save_checkpoint(
                model=model,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

            best_val_loss = val_loss

            print("\nSaved new best checkpoint.")

        else:
            print("\nValidation loss did not improve.")

        save_history(history)

        print("Training history saved.")

    print("\n" + "=" * 60)
    print("TRAINING EXPERIMENT FINISHED")
    print("=" * 60)

    print("Best checkpoint:", CHECKPOINT_PATH)
    print("Training history:", HISTORY_PATH)
    print(
        "\nThis is a limited CPU experiment. "
        "Evaluate the saved checkpoint on a representative "
        "test sample before reporting final accuracy."
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
