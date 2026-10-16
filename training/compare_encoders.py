import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.data_loader import (
    train_dataset,
    validation_dataset,
    DEVICE,
)
from preprocessing.snn_input import (
    rate_encode as original_encode,
    TIME_STEPS,
)
from preprocessing.snn_input_experiment import (
    rate_encode as experimental_encode,
)
from models.hybrid_snn import HybridSNN
from spikingjelly.activation_based import functional


SEED = 42
BATCH_SIZE = 4
TRAIN_BATCHES = 10
VAL_BATCHES = 10
EPOCHS = 2
LEARNING_RATE = 0.0001

torch.set_num_threads(2)
torch.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
)

val_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED + 1),
    num_workers=0,
)

# Cache the same batches for both encoder experiments.
train_batches = []
for i, batch in enumerate(train_loader):
    if i >= TRAIN_BATCHES:
        break
    train_batches.append(batch)

val_batches = []
for i, batch in enumerate(val_loader):
    if i >= VAL_BATCHES:
        break
    val_batches.append(batch)

print("Training images:", sum(len(y) for _, y in train_batches))
print("Validation images:", sum(len(y) for _, y in val_batches))


# Create one common starting model for a fair comparison.
base_model = HybridSNN(
    num_classes=38,
    time_steps=TIME_STEPS,
)
initial_weights = copy.deepcopy(base_model.state_dict())


def encode_without_changing_model_randomness(encoder, images, seed):
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    encoded = encoder(images, time_steps=TIME_STEPS)

    # Keep the model's random-number sequence comparable.
    torch.random.set_rng_state(rng_state)
    return encoded


def run_experiment(name, encoder):
    model = HybridSNN(
        num_classes=38,
        time_steps=TIME_STEPS,
    ).to(DEVICE)

    model.load_state_dict(initial_weights)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n===== {name} =====")

    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for batch_index, (images, labels) in enumerate(train_batches):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            spikes = encode_without_changing_model_randomness(
                encoder,
                images,
                SEED + epoch * 1000 + batch_index,
            ).to(DEVICE)

            optimizer.zero_grad()
            functional.reset_net(model)

            outputs = model(spikes)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            train_loss += loss.item()
            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            train_total += labels.size(0)

            functional.reset_net(model)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(val_batches):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                spikes = encode_without_changing_model_randomness(
                    encoder,
                    images,
                    SEED + 10000 + batch_index,
                ).to(DEVICE)

                functional.reset_net(model)
                outputs = model(spikes)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                val_total += labels.size(0)

                functional.reset_net(model)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss / len(train_batches):.4f} | "
            f"Train accuracy: "
            f"{100 * train_correct / max(train_total, 1):.2f}% | "
            f"Validation loss: "
            f"{val_loss / len(val_batches):.4f} | "
            f"Validation accuracy: "
            f"{100 * val_correct / max(val_total, 1):.2f}%"
        )

    functional.reset_net(model)


if __name__ == "__main__":
    try:
        run_experiment("ORIGINAL BINARY SPIKE ENCODER", original_encode)
        run_experiment("EXPERIMENTAL REPEATED-IMAGE ENCODER", experimental_encode)

        print("\nComparison complete.")
        print("No checkpoint was overwritten.")
        print("These small-sample results are diagnostic only.")

    finally:
        functional.reset_net(base_model)
