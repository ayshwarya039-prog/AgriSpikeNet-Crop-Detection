import torch

path = "checkpoints/best_model_weighted_snn.pth"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)

print("Checkpoint type:", type(checkpoint).__name__)

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys()))

    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if isinstance(state, dict):
        print("\nFirst 15 parameter entries:")
        for key, value in list(state.items())[:15]:
            if hasattr(value, "shape"):
                print(key, tuple(value.shape))
            else:
                print(key, type(value).__name__)
