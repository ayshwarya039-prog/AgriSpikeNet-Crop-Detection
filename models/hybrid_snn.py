
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

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
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

        # Keep classifier dimensions compatible with existing checkpoints.
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        # Learn how much each time step contributes to the final prediction.
        self.temporal_attention = nn.Linear(64, 1)

        self.last_spike_rates = []
        self.last_overall_spike_rate = 0.0

    def forward(self, spikes):
        if spikes.ndim != 5:
            raise ValueError("Expected input shape [T, B, C, H, W].")
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
            frame = F.interpolate(
                spikes[t].float(),
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )

            features = self.features(frame).flatten(start_dim=1)
            features = F.relu(self.pre_lif_norm(features))
            spiking_features = self.lif(features)

            temporal_features.append(spiking_features)
            spike_rates.append(
                spiking_features.detach().float().mean().item()
            )

        all_spikes = torch.stack(temporal_features, dim=0)

        self.last_spike_rates = spike_rates
        self.last_overall_spike_rate = (
            all_spikes.detach().float().mean().item()
        )

        # Attention-weighted fusion across time.
        scores = self.temporal_attention(all_spikes)
        weights = torch.softmax(scores, dim=0)
        fused_features = (weights * all_spikes).sum(dim=0)

        return self.classifier(fused_features)


if __name__ == "__main__":
    model = HybridSNN(num_classes=38, time_steps=10)
    model.eval()
    sample = torch.rand(10, 1, 3, 224, 224)

    try:
        with torch.no_grad():
            functional.reset_net(model)
            output = model(sample)

        print("Model test passed.")
        print("Input shape:", tuple(sample.shape))
        print("Output shape:", tuple(output.shape))
        assert output.shape == (1, 38)
    finally:
        functional.reset_net(model)
