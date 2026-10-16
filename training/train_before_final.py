
import sys
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


# ---------------- CONFIGURATION ----------------

BATCH_SIZE = 4
EPOCHS = 1

# Small limits are used to keep CPU training manageable.
MAX_TRAIN_BATCHES = 20
MAX_VAL_BATCHES = 5

LEARNING_RATE = 0.0001

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Keep the experiment separate from best_model.pth.
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model_experiment.pth"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(2)


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


train_loader = make_loader(train_dataset, True)
validation_loader = make_loader(validation_dataset, False)


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

        spikes, labels = convert_batch_to_spikes(
            images,
            labels,
            time_steps=TIME_STEPS,
        )
        spikes = spikes.to(DEVICE)

        functional.reset_net(model)

        if training:
            optimizer.zero_grad(set_to_none=True)

        try:
            with torch.set_grad_enabled(training):
                outputs = model(spikes)
                loss = nn.functional.cross_entropy(outputs, labels)

                if training:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

            batch_count = labels.size(0)

            total_loss += loss.item() * batch_count
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += batch_count
            batches_done += 1

            if (batch_index + 1) % 5 == 0:
                print(
                    f"  {'Train' if training else 'Validation'} "
                    f"batch {batch_index + 1} | "
                    f"loss={loss.item():.4f}"
                )

        finally:
            functional.reset_net(model)

        del images, labels, spikes, outputs, loss

    if total == 0:
        raise RuntimeError("No batches were processed.")

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy, batches_done


def main():
    print("=" * 60)
    print("AgriSpikeNet - Hybrid SNN Training")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Number of classes:", len(CLASS_NAMES))
    print("Batch size:", BATCH_SIZE)
    print("Time steps:", TIME_STEPS)
    print("Epochs:", EPOCHS)
    print("Training batch limit:", MAX_TRAIN_BATCHES)
    print("Validation batch limit:", MAX_VAL_BATCHES)
    print("Checkpoint:", CHECKPOINT_PATH)

    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=TIME_STEPS,
    ).to(DEVICE)

    # Load the experimental checkpoint when available.
    # Preserve its previous contents until a better result is measured.
    previous_checkpoint = None

    if CHECKPOINT_PATH.exists():
        print("\nLoading existing experimental checkpoint...")

        previous_checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        if "model_state_dict" not in previous_checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain model_state_dict. "
                "The file has not been modified."
            )

        if previous_checkpoint.get("class_names") != list(CLASS_NAMES):
            raise RuntimeError(
                "Checkpoint class names do not match the current dataset. "
                "The file has not been modified."
            )

        model.load_state_dict(
            previous_checkpoint["model_state_dict"],
            strict=True,
        )

        old_val_loss = previous_checkpoint.get("validation_loss")

        if old_val_loss is None:
            old_val_loss = float("inf")
        else:
            old_val_loss = float(old_val_loss)

        print("Existing checkpoint loaded.")
        print("Recorded validation loss:", old_val_loss)

    else:
        print("No experimental checkpoint found. Starting a new model.")
        old_val_loss = float("inf")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = old_val_loss
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========")

        train_loss, train_acc, train_count = run_epoch(
            model,
            train_loader,
            optimizer,
            training=True,
            max_batches=MAX_TRAIN_BATCHES,
        )

        val_loss, val_acc, val_count = run_epoch(
            model,
            validation_loader,
            optimizer,
            training=False,
            max_batches=MAX_VAL_BATCHES,
        )

        print("\nEpoch results:")
        print(f"Training batches: {train_count}")
        print(f"Validation batches: {val_count}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc * 100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc * 100:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "train_batches": train_count,
            "validation_batches": val_count,
        })

        # Save to a temporary file first.
        # Replace the experimental checkpoint only if validation loss
        # improves compared with its recorded value.
        if val_loss < best_val_loss:
            temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(CLASS_NAMES),
                    "class_names": list(CLASS_NAMES),
                    "time_steps": TIME_STEPS,
                    "epochs_completed": (
                        int(previous_checkpoint.get("epochs_completed", 0))
                        + epoch
                        if previous_checkpoint
                        else epoch
                    ),
                    "validation_loss": val_loss,
                },
                temporary_path,
            )

            temporary_path.replace(CHECKPOINT_PATH)
            best_val_loss = val_loss

            print("Saved improved experimental checkpoint.")

        else:
            print(
                "Checkpoint not replaced: validation loss "
                "did not improve over the recorded value."
            )

        with open(
            HISTORY_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys(),
            )
            writer.writeheader()
            writer.writerows(history)

    print("\nTraining run finished.")
    print("Experimental checkpoint:", CHECKPOINT_PATH)
    print("Training history:", HISTORY_PATH)
    print(
        "NOTE: This is a limited training run, not full-dataset training. "
        "The validation accuracy is based on only a small validation sample."
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        pass
