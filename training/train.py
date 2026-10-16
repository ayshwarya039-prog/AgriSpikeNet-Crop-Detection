
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
EPOCHS = 15

# CPU-friendly experiment limits
MAX_TRAIN_BATCHES = 1000
MAX_VAL_BATCHES = 200

LEARNING_RATE = 0.0001
RANDOM_SEED = 42

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Save this experiment separately.
# Your existing checkpoint will not be overwritten.
CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "best_model_weighted_snn_v2.pth"
)

HISTORY_PATH = (
    RESULTS_DIR / "training_history_weighted_snn_v2.csv"
)

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
        generator=generator,
    )


train_loader = make_loader(
    train_dataset,
    shuffle=True,
)

validation_loader = make_loader(
    validation_dataset,
    shuffle=False,
)

# --------------------------------------------------
# INPUT ENCODING
# --------------------------------------------------

def encode_images(images, labels):
    """
    Convert normalized images using the project's
    deterministic rate encoder.

    Output shape:
    [time_steps, batch_size, channels, height, width]

    Note: the current encoder repeats continuous image
    rates across time; it does not create binary spikes.
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

def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    training,
    max_batches,
):
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0
    batches_done = 0

    context = (
        torch.enable_grad
        if training
        else torch.no_grad
    )

    for batch_index, (images, labels) in enumerate(loader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        encoded, labels = encode_images(images, labels)

        # Reset LIF state before each independent batch.
        functional.reset_net(model)

        try:
            if training:
                optimizer.zero_grad(set_to_none=True)

            with context():
                outputs = model(encoded)
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

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size
            batches_done += 1

            if (batch_index + 1) % 100 == 0:
                phase = (
                    "Training"
                    if training
                    else "Validation"
                )

                print(
                    f"{phase} batch "
                    f"{batch_index + 1}/{max_batches}: "
                    f"loss={loss.item():.4f}"
                )

        finally:
            # Avoid carrying neuron state into another batch.
            functional.reset_net(model)

        del images, labels, encoded, outputs, loss

    if total == 0:
        raise RuntimeError(
            "No batches were processed. "
            "Please check the dataset and data loader."
        )

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy, batches_done


# --------------------------------------------------
# SAVE BEST CHECKPOINT
# --------------------------------------------------

def save_checkpoint(
    model,
    epoch,
    val_loss,
    val_accuracy,
):
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
                "loss": "class_weighted_cross_entropy",
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

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "train_batches",
        "validation_batches",
    ]

    with open(
        temporary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
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
    print("=" * 65)
    print("AgriSpikeNet - Weighted Hybrid SNN V2")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Classes:", len(CLASS_NAMES))
    print("Batch size:", BATCH_SIZE)
    print("Time steps:", TIME_STEPS)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Training batches per epoch:", MAX_TRAIN_BATCHES)
    print("Validation batches per epoch:", MAX_VAL_BATCHES)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)

    # Fresh model for this separate experiment.
    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=TIME_STEPS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Compute class weights using training labels only.
    class_counts = torch.bincount(
        torch.tensor(
            train_dataset.targets,
            dtype=torch.long,
        ),
        minlength=len(CLASS_NAMES),
    ).float()

    if torch.any(class_counts == 0):
        raise RuntimeError(
            "At least one class has no training images."
        )

    class_weights = (
        class_counts.sum()
        / (
            len(CLASS_NAMES) * class_counts
        )
    )

    class_weights = class_weights.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    print("\nClass-weighted loss enabled.")
    print("Number of class weights:", len(class_weights))
    print(
        "Training images per epoch:",
        min(
            len(train_dataset),
            MAX_TRAIN_BATCHES * BATCH_SIZE,
        ),
    )
    print(
        "Validation images per epoch:",
        min(
            len(validation_dataset),
            MAX_VAL_BATCHES * BATCH_SIZE,
        ),
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):

        print(
            f"\n{'=' * 20} "
            f"EPOCH {epoch}/{EPOCHS} "
            f"{'=' * 20}"
        )

        train_loss, train_accuracy, train_count = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            training=True,
            max_batches=MAX_TRAIN_BATCHES,
        )

        val_loss, val_accuracy, val_count = run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=None,
            criterion=criterion,
            training=False,
            max_batches=MAX_VAL_BATCHES,
        )

        print("\nEpoch results")
        print("-" * 45)
        print("Training batches:", train_count)
        print("Validation batches:", val_count)
        print(f"Training loss: {train_loss:.4f}")
        print(
            f"Training accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )
        print(f"Validation loss: {val_loss:.4f}")
        print(
            f"Validation accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

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

            print("\nSaved new best V2 checkpoint.")

        else:
            print(
                "\nValidation loss did not improve."
            )

        save_history(history)
        print("Training history saved.")

    print("\n" + "=" * 65)
    print("V2 TRAINING EXPERIMENT FINISHED")
    print("=" * 65)

    print("Best checkpoint:", CHECKPOINT_PATH)
    print("Training history:", HISTORY_PATH)
    print(f"Best validation loss: {best_val_loss:.4f}")

    print(
        "\nThis is a limited CPU training experiment. "
        "Evaluate the saved checkpoint on validation data "
        "before reporting final performance."
    )


if __name__ == "__main__":
    main()
