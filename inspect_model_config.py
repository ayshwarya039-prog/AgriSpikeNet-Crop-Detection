import torch

path = "checkpoints/best_model_weighted_snn.pth"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("Saved model configuration:")
print(ckpt.get("model_config"))

print("\nSaved validation loss:", ckpt.get("validation_loss"))
print("Saved validation accuracy:", ckpt.get("validation_accuracy"))
print("Saved epochs completed:", ckpt.get("epochs_completed"))
print("Saved time steps:", ckpt.get("time_steps"))
