
import os
import torch

folder = "checkpoints"

for filename in os.listdir(folder):
    if not filename.endswith(".pth"):
        continue

    path = os.path.join(folder, filename)

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            print("\n" + "=" * 50)
            print("File:", filename)

            for key in [
                "epoch",
                "epochs",
                "time_steps",
                "val_accuracy",
                "validation_accuracy",
                "best_val_accuracy",
                "val_loss",
                "validation_loss",
            ]:
                if key in checkpoint:
                    print(f"{key}: {checkpoint[key]}")

            state = checkpoint.get("model_state_dict")
            if state is not None:
                print("Model weights: found")
                print("Number of weight entries:", len(state))
            else:
                print("Model weights: check checkpoint format")

    except Exception as error:
        print("Could not inspect:", filename)
        print("Reason:", error)

