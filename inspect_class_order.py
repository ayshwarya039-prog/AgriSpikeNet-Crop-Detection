import torch
from preprocessing import data_loader

path = "checkpoints/best_model_weighted_snn.pth"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

saved_classes = ckpt.get("class_names", [])
current_classes = list(data_loader.CLASS_NAMES)

print("Checkpoint saved validation accuracy:",
      ckpt.get("validation_accuracy"))
print("Checkpoint time steps:", ckpt.get("time_steps"))
print("Checkpoint class count:", len(saved_classes))
print("Current dataset class count:", len(current_classes))
print("Class order matches:", saved_classes == current_classes)

if saved_classes != current_classes:
    print("\nFirst class-order differences:")
    for i, (old, new) in enumerate(zip(saved_classes, current_classes)):
        if old != new:
            print(f"Index {i}: checkpoint={old!r} | current={new!r}")
            if i >= 10:
                break
