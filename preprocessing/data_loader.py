# ============================================================
# AgriSpikeNet - Multi-Crop DataLoader
# ============================================================

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

CLASS_NAMES_FILE = PROJECT_DIR / "class_names.json"


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. PRINT HEADER
# ============================================================

print("\n" + "=" * 75)
print("          AgriSpikeNet MULTI-CROP DATA LOADER")
print("=" * 75)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nDataset directory:")
print(DATASET_DIR)

print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():
    print("GPU:")
    print(torch.cuda.get_device_name(0))
else:
    print("GPU not available. Using CPU.")


# ============================================================
# 5. CHECK DATASET DIRECTORIES
# ============================================================

required_directories = [
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR
]


print("\n" + "=" * 75)
print("CHECKING DATASET DIRECTORIES")
print("=" * 75)


for directory in required_directories:

    if not directory.exists():

        raise FileNotFoundError(
            f"\nDataset directory not found:\n{directory}\n\n"
            "Please run split_dataset.py first."
        )

    print(f"\n✓ Found: {directory}")


# ============================================================
# 6. IMAGE TRANSFORMS
# ============================================================

# Training transformations
# Includes light augmentation to improve generalization.

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# Validation transformations
# No random augmentation.

validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# Test transformations
# Same deterministic preprocessing as validation.

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 7. LOAD DATASETS
# ============================================================

print("\n" + "=" * 75)
print("LOADING DATASETS")
print("=" * 75)


print("\nLoading training dataset...")

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)

print(
    f"✓ Training dataset loaded: "
    f"{len(train_dataset)} images"
)


print("\nLoading validation dataset...")

validation_dataset = datasets.ImageFolder(
    root=str(VALIDATION_DIR),
    transform=validation_transform
)

print(
    f"✓ Validation dataset loaded: "
    f"{len(validation_dataset)} images"
)


print("\nLoading test dataset...")

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=test_transform
)

print(
    f"✓ Test dataset loaded: "
    f"{len(test_dataset)} images"
)


# ============================================================
# 8. CHECK CLASS CONSISTENCY
# ============================================================

print("\n" + "=" * 75)
print("CHECKING CLASS INFORMATION")
print("=" * 75)


train_classes = train_dataset.classes

validation_classes = validation_dataset.classes

test_classes = test_dataset.classes


print(
    f"\nTraining classes   : {len(train_classes)}"
)

print(
    f"Validation classes : {len(validation_classes)}"
)

print(
    f"Test classes       : {len(test_classes)}"
)


if train_classes != validation_classes:

    raise RuntimeError(
        "\nERROR: Training and validation classes do not match."
    )


if train_classes != test_classes:

    raise RuntimeError(
        "\nERROR: Training and test classes do not match."
    )


CLASS_NAMES = train_classes

NUM_CLASSES = len(CLASS_NAMES)


print(
    f"\n✓ All dataset splits contain "
    f"the same {NUM_CLASSES} classes."
)


# ============================================================
# 9. PRINT CLASS LIST
# ============================================================

print("\n" + "=" * 75)
print("CLASS LIST")
print("=" * 75)


for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{index:2d} -> {class_name}"
    )


# ============================================================
# 10. SAVE CLASS NAMES
# ============================================================

class_mapping = {
    str(index): class_name
    for index, class_name
    in enumerate(CLASS_NAMES)
}


with open(
    CLASS_NAMES_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_mapping,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    f"\n✓ Class names saved to:"
    f"\n{CLASS_NAMES_FILE}"
)


# ============================================================
# 11. CREATE DATA LOADERS
# ============================================================

print("\n" + "=" * 75)
print("CREATING PYTORCH DATALOADERS")
print("=" * 75)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


print(
    "\n✓ Training DataLoader created"
)

print(
    "✓ Validation DataLoader created"
)

print(
    "✓ Test DataLoader created"
)


# ============================================================
# 12. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("DATASET SUMMARY")
print("=" * 75)


print(
    f"\nNumber of classes : {NUM_CLASSES}"
)

print(
    f"Training images   : {len(train_dataset)}"
)

print(
    f"Validation images : {len(validation_dataset)}"
)

print(
    f"Test images       : {len(test_dataset)}"
)

print(
    f"Image size        : {IMAGE_SIZE} x {IMAGE_SIZE}"
)

print(
    f"Batch size        : {BATCH_SIZE}"
)

print(
    f"Number of workers : {NUM_WORKERS}"
)


# ============================================================
# 13. CHECK FIRST TRAINING BATCH
# ============================================================

print("\n" + "=" * 75)
print("CHECKING FIRST TRAINING BATCH")
print("=" * 75)


try:

    images, labels = next(
        iter(train_loader)
    )

    print(
        "\nImage tensor shape:"
    )

    print(
        images.shape
    )

    print(
        "\nLabel tensor shape:"
    )

    print(
        labels.shape
    )

    print(
        "\nImage tensor data type:"
    )

    print(
        images.dtype
    )

    print(
        "\nLabel tensor data type:"
    )

    print(
        labels.dtype
    )

    print(
        "\nMinimum pixel value:"
    )

    print(
        images.min().item()
    )

    print(
        "\nMaximum pixel value:"
    )

    print(
        images.max().item()
    )

    print(
        "\n✓ First training batch loaded successfully."
    )

except Exception as error:

    raise RuntimeError(
        "\nCould not load the first training batch."
        f"\nError: {error}"
    )


# ============================================================
# 14. VERIFY LABEL RANGE
# ============================================================

print("\n" + "=" * 75)
print("VERIFYING LABELS")
print("=" * 75)


minimum_label = labels.min().item()

maximum_label = labels.max().item()


print(
    f"\nMinimum label : {minimum_label}"
)

print(
    f"Maximum label : {maximum_label}"
)


if minimum_label < 0:

    raise RuntimeError(
        "\nInvalid negative label detected."
    )


if maximum_label >= NUM_CLASSES:

    raise RuntimeError(
        "\nInvalid label detected."
    )


print(
    "\n✓ Labels are valid."
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("AgriSpikeNet DATA LOADER READY")
print("=" * 75)


print(
    "\nThe following objects are available:"
)

print(
    "  ✓ train_dataset"
)

print(
    "  ✓ validation_dataset"
)

print(
    "  ✓ test_dataset"
)

print(
    "  ✓ train_loader"
)

print(
    "  ✓ validation_loader"
)

print(
    "  ✓ test_loader"
)

print(
    "  ✓ CLASS_NAMES"
)

print(
    "  ✓ NUM_CLASSES"
)

print(
    "  ✓ DEVICE"
)


print(
    "\nDataset is ready for SNN input conversion."
)

print(
    "\nNext stage:"
)

print(
    "preprocessing/snn_input.py"
)

print(
    "\n" + "=" * 75
)
