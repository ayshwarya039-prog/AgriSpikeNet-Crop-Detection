
import sys
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from spikingjelly.activation_based import functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import train_dataset, NUM_CLASSES, DEVICE
from preprocessing.snn_input import convert_batch_to_spikes
from models.hybrid_snn import HybridSNN

# Keep this experiment small and separate from normal training.
NUM_CLASSES_TO_TEST = 8
IMAGES_PER_CLASS = 4
TRAINING_STEPS = 20
LEARNING_RATE = 0.001

random.seed(42)
torch.manual_seed(42)

print("\nPreparing a balanced diagnostic dataset...")

# Select 4 images from each of the first 8 classes.
selected_indices = []
class_counts = {}

for index, (_, label) in enumerate(train_dataset):
    label = int(label)

    if label not in class_counts:
        class_counts[label] = 0

    if label < NUM_CLASSES_TO_TEST:
        if class_counts[label] < IMAGES_PER_CLASS:
            selected_indices.append(index)
            class_counts[label] += 1

    if len(selected_indices) == NUM_CLASSES_TO_TEST * IMAGES_PER_CLASS:
        break

if len(selected_indices) != NUM_CLASSES_TO_TEST * IMAGES_PER_CLASS:
    raise RuntimeError("Could not collect enough images for every selected class.")

images = torch.stack([
    train_dataset[index][0] for index in selected_indices
]).to(DEVICE)

labels = torch.tensor([
    int(train_dataset[index][1]) for index in selected_indices
], dtype=torch.long, device=DEVICE)

print("Images:", len(images))
print("Selected classes:", NUM_CLASSES_TO_TEST)

# Encode only once, so the spike inputs remain fixed during this test.
spikes, labels = convert_batch_to_spikes(images, labels)
spikes = spikes.to(DEVICE)
labels = labels.to(DEVICE)

print("Fixed spike tensor shape:", tuple(spikes.shape))

model = HybridSNN(
    num_classes=NUM_CLASSES,
    time_steps=spikes.shape[0]
).to(DEVICE)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("\nStarting diagnostic training...")

for step in range(1, TRAINING_STEPS + 1):
    model.train()
    optimizer.zero_grad()

    functional.reset_net(model)
    outputs = model(spikes)
    loss = criterion(outputs, labels)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    functional.reset_net(model)

    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == labels).float().mean().item() * 100

    if step == 1 or step % 5 == 0:
        print(
            f"Step {step:02d}/{TRAINING_STEPS} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

print("\nDiagnostic experiment completed.")
print("This measures learning on the selected training images only.")
print("It does NOT measure generalization to unseen leaves.")
print("No existing checkpoint was modified.")
