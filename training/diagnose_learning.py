
import sys
from pathlib import Path
import torch
import torch.nn as nn
import io
from contextlib import redirect_stdout

# Allow Python to find the project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import (
    train_dataset,
    CLASS_NAMES,
    DEVICE,
)
from preprocessing.snn_input import (
    convert_batch_to_spikes,
    TIME_STEPS,
)
from models.hybrid_snn import HybridSNN
from spikingjelly.activation_based import functional


def main():
    torch.set_num_threads(2)

    print("\nStarting Hybrid SNN learning diagnostic...")
    print("Device:", DEVICE)

    # Select four images from four different classes
    selected_images = []
    selected_labels = []
    found_labels = set()

    for index in range(len(train_dataset)):
        image, label = train_dataset[index]
        label = int(label)

        if label not in found_labels:
            selected_images.append(image)
            selected_labels.append(label)
            found_labels.add(label)

        if len(selected_images) == 4:
            break

    if len(selected_images) < 4:
        print("ERROR: Could not find four different classes.")
        return

    images = torch.stack(selected_images).to(DEVICE)
    labels = torch.tensor(
        selected_labels,
        dtype=torch.long,
        device=DEVICE,
    )

    print("\nSelected classes:")
    for label in selected_labels:
        print("-", CLASS_NAMES[label])

    # Convert the images into spikes once
    encoded = convert_batch_to_spikes(
        images,
        labels,
        time_steps=TIME_STEPS,
    )

    # Handle encoders that return either a tensor or a tuple
    if isinstance(encoded, tuple):
        spikes = encoded[0]
    else:
        spikes = encoded

    spikes = spikes.to(DEVICE)

    print("Spike tensor shape:", tuple(spikes.shape))

    # Create a new model for this test
    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=TIME_STEPS,
    )

    # Confirm that the model is actually a PyTorch model
    if not isinstance(model, nn.Module):
        print("ERROR: HybridSNN did not return a PyTorch model.")
        print("Returned type:", type(model))
        return

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("\nTesting learning over 15 steps.")
    print("No checkpoint will be saved or overwritten.\n")

    for step in range(1, 16):
        model.train()
        optimizer.zero_grad()

        functional.reset_net(model)

        # Hide the model's diagnostic prints
        with redirect_stdout(io.StringIO()):
            outputs = model(spikes)
            loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        predictions = outputs.argmax(dim=1)
        accuracy = (
            (predictions == labels).float().mean().item() * 100
        )

        functional.reset_net(model)

        if step in (1, 5, 10, 15):
            print(
                f"Step {step:2d}/15 | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy:.1f}%"
            )

    print("\nDiagnostic finished.")
    print("No checkpoint was saved or changed.")


if __name__ == "__main__":
    main()
