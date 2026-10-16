from pathlib import Path
import random
import shutil
# ============================================================
# AgriSpikeNet - PlantVillage Dataset Preparation
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# The 4 classes used in your project
CLASSES = {
    "Tomato___Early_blight": "Early_Blight",
    "Tomato___healthy": "Healthy",
    "Tomato___Late_blight": "Late_Blight",
    "Tomato___Septoria_leaf_spot": "Leaf_Spot",
}

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


# ============================================================
# Create folders
# ============================================================

def create_folders():

    print("\nCreating dataset folders...")

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        for class_name in CLASSES.values():

            folder_path = folder / class_name

            folder_path.mkdir(
                parents=True,
                exist_ok=True
            )

    print("Folders created successfully.")


# ============================================================
# Find PlantVillage dataset
# ============================================================

def find_source_dataset():

    possible_locations = [

        BASE_DIR / "PlantVillage",

        BASE_DIR / "plantvillage",

        BASE_DIR / "PlantVillage-Dataset",

        BASE_DIR / "plantvillage-dataset",

        Path.home() / "Downloads" / "PlantVillage",

        Path.home() / "Downloads" / "plantvillage",

        Path.home() / "Downloads" / "PlantVillage-Dataset",

        Path.home() / "Downloads" / "plantvillage-dataset",
    ]

    print("\nSearching for PlantVillage dataset...")

    for location in possible_locations:

        print("Checking:", location)

        if location.exists():

            print("\nPlantVillage found at:")
            print(location)

            return location

    return None


# ============================================================
# Get images
# ============================================================

def get_images(folder):

    extensions = [
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.JPEG",
        "*.png",
        "*.PNG"
    ]

    images = []

    for extension in extensions:

        images.extend(folder.glob(extension))

    return images


# ============================================================
# Clear old generated dataset
# ============================================================

def clear_old_dataset():

    print("\nRemoving old train/validation/test images...")

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        if folder.exists():

            for class_name in CLASSES.values():

                class_folder = folder / class_name

                if class_folder.exists():

                    for file in class_folder.iterdir():

                        if file.is_file():

                            file.unlink()

    print("Old dataset cleared.")


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(source_dir):

    print("\n" + "=" * 60)
    print("Preparing AgriSpikeNet Dataset")
    print("=" * 60)

    total_images = 0

    for original_class, new_class in CLASSES.items():

        source_class = source_dir / original_class

        print("\n--------------------------------------------")
        print("Class:", original_class)
        print("--------------------------------------------")

        if not source_class.exists():

            print("WARNING: Folder not found:")
            print(source_class)

            continue

        images = get_images(source_class)

        if len(images) == 0:

            print("WARNING: No images found.")

            continue

        # Shuffle images
        random.shuffle(images)

        total = len(images)

        # Calculate split
        train_count = int(total * TRAIN_RATIO)

        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]

        val_images = images[
            train_count:
            train_count + val_count
        ]

        test_images = images[
            train_count + val_count:
        ]

        print("Total images :", total)
        print("Training     :", len(train_images))
        print("Validation   :", len(val_images))
        print("Testing      :", len(test_images))

        # ====================================================
        # Training
        # ====================================================

        for i, image in enumerate(train_images):

            destination = (
                TRAIN_DIR /
                new_class /
                f"{new_class}_{i:05d}{image.suffix}"
            )

            shutil.copy2(
                image,
                destination
            )

        # ====================================================
        # Validation
        # ====================================================

        for i, image in enumerate(val_images):

            destination = (
                VAL_DIR /
                new_class /
                f"{new_class}_{i:05d}{image.suffix}"
            )

            shutil.copy2(
                image,
                destination
            )

        # ====================================================
        # Testing
        # ====================================================

        for i, image in enumerate(test_images):

            destination = (
                TEST_DIR /
                new_class /
                f"{new_class}_{i:05d}{image.suffix}"
            )

            shutil.copy2(
                image,
                destination
            )

        total_images += total

    print("\n" + "=" * 60)
    print("DATASET PREPARATION FINISHED")
    print("=" * 60)

    print("\nTotal images processed:", total_images)

    print("\nDataset location:")
    print(DATASET_DIR)

    print("\nYour final structure will be:")

    print("""
dataset/
│
├── train/
│   ├── Early_Blight/
│   ├── Healthy/
│   ├── Late_Blight/
│   └── Leaf_Spot/
│
├── validation/
│   ├── Early_Blight/
│   ├── Healthy/
│   ├── Late_Blight/
│   └── Leaf_Spot/
│
└── test/
    ├── Early_Blight/
    ├── Healthy/
    ├── Late_Blight/
    └── Leaf_Spot/
""")

    print("Next step:")
    print("python preprocessing\\data_loader.py")


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("AgriSpikeNet - Dataset Preparation")
    print("=" * 60)

    # Create folders
    create_folders()

    # Find PlantVillage
    source_dir = find_source_dataset()

    if source_dir is None:

        print("\n" + "=" * 60)
        print("PLANTVILLAGE DATASET NOT FOUND")
        print("=" * 60)

        print("""
Please download the PlantVillage dataset.

Then place the dataset folder here:

AgriSpikeNet Crop Detection/
│
├── PlantVillage/
│   ├── Tomato___Early_blight/
│   ├── Tomato___healthy/
│   ├── Tomato___Late_blight/
│   └── Tomato___Septoria_leaf_spot/
│
├── dataset/
├── preprocessing/
├── models/
└── download_dataset.py

Then run:

python download_dataset.py
""")

        return

    # Remove previous generated files
    clear_old_dataset()

    # Create new dataset
    prepare_dataset(source_dir)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()
