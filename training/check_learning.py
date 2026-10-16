import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from preprocessing.data_loader import (
    train_loader,
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
    print("=" * 50)
    print("AgriSpikeNet - Learning Diagnostic")
    print("=" * 50)

    images, labels = next(iter(train_loader))

    images = images[:1].to(DEVICE)
    labels = labels[:1].to(DEVICE)

    spikes, labels = convert_batch_to_spikes(
        images,
        labels,
        TIME_STEPS,
    )

    print("\n--- Dataset information ---")
    print("Device:", DEVICE)
    print("Number of classes:", len(CLASS_NAMES))
    print("Image shape:", tuple(images.shape))
    print("Spike shape:", tuple(spikes.shape))
    print("Actual class:", CLASS_NAMES[labels.item()])

    input_rate = spikes.float().mean().item()

    print("\n--- Input spike activity ---")
    print("Input spike rate:", round(input_rate, 4))
    print("Input spike percentage:", round(input_rate * 100, 2), "%")
    print(
        "Binary spikes:",
        bool(torch.all((spikes == 0) | (spikes == 1)).item()),
    )

    model = HybridSNN(
        num_classes=len(CLASS_NAMES),
        time_steps=TIME_STEPS,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        functional.reset_net(model)

        outputs = model(spikes)

        print("\n--- LIF spike activity ---")
        print(
            "Spike rates by time step:",
            [
                round(rate, 4)
                for rate in model.last_spike_rates
            ],
        )
        print(
            "Overall LIF spike rate:",
            round(model.last_overall_spike_rate, 4),
        )
        print(
            "Overall LIF spike percentage:",
            round(model.last_overall_spike_rate * 100, 2),
            "%",
        )

        loss = criterion(outputs, labels)
        loss.backward()

        print("\n--- Learning checks ---")
        print("Output shape:", tuple(outputs.shape))
        print("Loss:", round(loss.item(), 4))
        print("Loss is finite:", bool(torch.isfinite(loss).item()))

        gradient_count = 0
        parameter_count = 0

        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter_count += 1

                if (
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all().item()
                    and parameter.grad.abs().sum().item() > 0
                ):
                    gradient_count += 1

        print(
            "Parameters with nonzero gradients:",
            gradient_count,
            "/",
            parameter_count,
        )

        predicted_index = outputs.detach().argmax(dim=1).item()
        print("Predicted class:", CLASS_NAMES[predicted_index])

        if (
            torch.isfinite(loss).item()
            and gradient_count > 0
        ):
            optimizer.step()
            print("Optimizer step completed.")
            print("Basic learning diagnostic completed.")
        else:
            print(
                "WARNING: Loss or gradients need checking "
                "before full training."
            )

    finally:
        functional.reset_net(model)


if __name__ == "__main__":
    main()