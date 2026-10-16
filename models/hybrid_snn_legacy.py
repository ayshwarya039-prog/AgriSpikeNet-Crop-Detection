
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional


class HybridSNN(nn.Module):
    """
    Hybrid CNN-SNN for AgriSpikeNet.

    Input:  [T, B, C, H, W]
    Output: [B, num_classes]
    """

    def __init__(self, num_classes=38, time_steps=10):
        super().__init__()

        self.time_steps = time_steps

        # CNN spatial feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.pre_lif_norm = nn.LayerNorm(64)

        self.lif = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            detach_reset=True,
        )

        # Classify the average spike activity over time
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        self.last_spike_rates = []
        self.last_overall_spike_rate = 0.0

    def forward(self, spikes):
        if spikes.ndim != 5:
            raise ValueError(
                "Expected input shape [T, B, C, H, W]."
            )

        if spikes.shape[0] != self.time_steps:
            raise ValueError(
                f"Expected {self.time_steps} time steps, "
                f"received {spikes.shape[0]}."
            )

        if spikes.shape[2] != 3:
            raise ValueError("Expected 3 input channels (RGB).")

        temporal_features = []
        spike_rates = []

        for t in range(self.time_steps):
            frame = spikes[t].float()

            # Reduce CPU computation
            frame = F.interpolate(
                frame,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )

            features = self.features(frame)
            features = features.flatten(start_dim=1)
            features = self.pre_lif_norm(features)
            features = F.relu(features)

            spiking_features = self.lif(features)

            spike_rates.append(
                spiking_features.detach().float().mean().item()
            )
            temporal_features.append(spiking_features)

        all_spikes = torch.stack(temporal_features, dim=0)

        self.last_spike_rates = spike_rates
        self.last_overall_spike_rate = (
            all_spikes.detach().float().mean().item()
        )

        # Average spike activity across time
        fused_features = all_spikes.mean(dim=0)
        outputs = self.classifier(fused_features)

        return outputs


if __name__ == "__main__":
    model = HybridSNN(num_classes=38, time_steps=10)
    model.eval()

    sample_spikes = torch.randint(
        0, 2, (10, 1, 3, 224, 224)
    ).float()

    try:
        with torch.no_grad():
            functional.reset_net(model)
            output = model(sample_spikes)

        print("Hybrid SNN model test completed.")
        print("Input shape:", tuple(sample_spikes.shape))
        print("Output shape:", tuple(output.shape))
        print("Number of classes:", output.shape[1])
        print(
            "LIF spike rates:",
            [round(rate, 4) for rate in model.last_spike_rates],
        )
        print(
            "Overall spike rate:",
            round(model.last_overall_spike_rate, 4),
        )

        assert output.shape == (1, 38)

    finally:
        functional.reset_net(model)
