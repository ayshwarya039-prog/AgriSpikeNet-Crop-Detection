import random
import shutil
import os
import time
from pathlib import Path


# ============================================================
# AgriSpikeNet - Multi-Crop Dataset Splitter
# Windows Safe Version
# ============================================================

print("\n" + "=" * 80)
print("              AgriSpikeNet MULTI-CROP DATASET SPLITTER")
print("=" * 80)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Original PlantVillage dataset
SOURCE_DIR = PROJECT_DIR / "PlantVillage"

# Generated dataset
DATASET_DIR = PROJECT_DIR / "dataset"

# Train / Validation / Test directories
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


print("\nProject directory:")
print(PROJECT_DIR)

print("\nOriginal PlantVillage dataset:")
print(SOURCE_DIR)

print("\nGenerated dataset:")
print(DATASET_DIR)


# ============================================================
# 2. CHECK SOURCE DATASET
# ============================================================

if not SOURCE_DIR.exists():
    print("\n" + "=" * 80)
    print("ERROR")
    print("=" * 80)

    print("\nPlantVillage folder was not found.")

    print("\nExpected location:")
    print(SOURCE_DIR)

    print("\nYour project should look like:")

    print("""
AgriSpikeNet Crop Detection/
│
├── PlantVillage/
│   ├── Apple___Apple_scab/
│   ├── Apple___healthy/
│   ├── Cherry_(including_sour)___healthy/
│   ├── Corn_(maize)___Common_rust_
│   ├── ...
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── preprocessing/
├── models/
├── training/
└── app/
""")

    raise SystemExit(1)


# ============================================================
# 3. SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# 4. CHECK SPLIT RATIOS
# ============================================================

ratio_sum = (
    TRAIN_RATIO
    + VALIDATION_RATIO
    + TEST_RATIO
)

if abs(ratio_sum - 1.0) > 0.0001:

    raise ValueError(
        "\nTrain + Validation + Test ratios must equal 1.0"
    )


# ============================================================
# 5. SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def is_image_file(file_path):
    """
    Check whether a file is a supported image.
    """

    return (
        file_path.is_file()
        and file_path.suffix.lower()
        in IMAGE_EXTENSIONS
    )


# ------------------------------------------------------------


def force_remove_file(file_path, retries=5):
    """
    Safely remove a file on Windows.

    Tries to remove read-only attributes and retries
    if Windows temporarily locks the file.
    """

    if not file_path.exists():
        return True

    for attempt in range(1, retries + 1):

        try:

            # Remove read-only attribute
            try:
                os.chmod(file_path, 0o666)
            except Exception:
                pass

            file_path.unlink()

            return True

        except PermissionError:

            if attempt < retries:
                time.sleep(1)

            else:
                return False

        except FileNotFoundError:

            return True

        except Exception:

            if attempt < retries:
                time.sleep(1)

            else:
                return False

    return False


# ------------------------------------------------------------


def force_remove_directory(directory, retries=5):
    """
    Safely remove a directory.

    This function attempts to remove files first,
    then removes subdirectories.
    """

    if not directory.exists():
        return True

    for attempt in range(1, retries + 1):

        try:

            # Remove all files and folders inside
            for item in directory.iterdir():

                if item.is_file():

                    if not force_remove_file(item):
                        raise PermissionError(
                            f"Could not remove file: {item}"
                        )

                elif item.is_dir():

                    if not force_remove_directory(item):
                        raise PermissionError(
                            f"Could not remove directory: {item}"
                        )

            # Remove empty directory
            try:
                os.chmod(directory, 0o777)
            except Exception:
                pass

            directory.rmdir()

            return True

        except PermissionError:

            if attempt < retries:
                time.sleep(1)

            else:
                return False

        except FileNotFoundError:

            return True

        except OSError:

            if attempt < retries:
                time.sleep(1)

            else:
                return False

    return False


# ------------------------------------------------------------


def clear_directory(directory):
    """
    Clear a directory safely.

    If Windows/OneDrive has locked something,
    a useful error message is displayed.
    """

    if not directory.exists():
        return True

    print(f"\nClearing:")
    print(directory)

    items = list(directory.iterdir())

    if len(items) == 0:

        print("  Already empty.")

        return True

    failed_items = []

    for item in items:

        if item.is_dir():

            success = force_remove_directory(item)

        else:

            success = force_remove_file(item)

        if not success:

            failed_items.append(item)

    if failed_items:

        print("\n" + "=" * 80)
        print("PERMISSION ERROR")
        print("=" * 80)

        print(
            "\nWindows/OneDrive is currently locking "
            "the following item(s):"
        )

        for item in failed_items[:20]:
            print(f"  {item}")

        print("\nPossible reasons:")

        print("  1. OneDrive is synchronizing the folder.")
        print("  2. The image is open in another program.")
        print("  3. Windows Explorer is using the folder.")
        print("  4. Another Python process is using the files.")
        print("  5. VS Code has a file operation in progress.")

        print("\nRecommended solution:")

        print("  • Close image viewers.")
        print("  • Close File Explorer windows.")
        print("  • Pause OneDrive synchronization temporarily.")
        print("  • Close VS Code and reopen it if necessary.")
        print("  • Then run this script again.")

        return False

    print("  Cleared successfully.")

    return True


# ------------------------------------------------------------


def copy_images(images, destination_directory):
    """
    Copy images into a destination directory.
    """

    destination_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    copied = 0

    for image in images:

        destination = (
            destination_directory
            / image.name
        )

        try:

            shutil.copy2(
                image,
                destination
            )

            copied += 1

        except PermissionError:

            print(
                f"\nWARNING: Permission denied while copying:"
            )

            print(image)

            print(
                "\nPlease make sure OneDrive is not "
                "locking the dataset."
            )

            raise

        except Exception as error:

            print(
                f"\nERROR while copying:"
            )

            print(image)

            print(
                f"Reason: {error}"
            )

            raise

    return copied


# ============================================================
# 7. FIND ALL CLASSES
# ============================================================

class_directories = sorted(
    [
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ],
    key=lambda x: x.name.lower()
)


if len(class_directories) == 0:

    raise RuntimeError(
        "\nNo class folders found inside PlantVillage."
    )


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print(
    f"\nNumber of classes found: "
    f"{len(class_directories)}"
)

print("\nClasses:")

for index, folder in enumerate(
    class_directories,
    start=1
):

    print(
        f"{index:2d}. {folder.name}"
    )


# ============================================================
# 9. CREATE DATASET DIRECTORIES
# ============================================================

print("\n" + "=" * 80)
print("PREPARING GENERATED DATASET")
print("=" * 80)


DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 10. CLEAR OLD DATA
# ============================================================

print("\n" + "=" * 80)
print("CLEARING OLD GENERATED DATA")
print("=" * 80)


train_clear = clear_directory(
    TRAIN_DIR
)

if not train_clear:

    raise SystemExit(
        "\nDataset splitting stopped because "
        "dataset/train could not be cleared."
    )


validation_clear = clear_directory(
    VALIDATION_DIR
)

if not validation_clear:

    raise SystemExit(
        "\nDataset splitting stopped because "
        "dataset/validation could not be cleared."
    )


test_clear = clear_directory(
    TEST_DIR
)

if not test_clear:

    raise SystemExit(
        "\nDataset splitting stopped because "
        "dataset/test could not be cleared."
    )


# ============================================================
# 11. PROCESS DATASET
# ============================================================

total_train = 0
total_validation = 0
total_test = 0

total_original = 0

processed_classes = 0

skipped_classes = 0


print("\n" + "=" * 80)
print("SPLITTING DATASET")
print("=" * 80)


for class_index, class_folder in enumerate(
    class_directories,
    start=1
):

    class_name = class_folder.name

    print("\n" + "-" * 80)

    print(
        f"[{class_index}/{len(class_directories)}] "
        f"Processing: {class_name}"
    )

    print("-" * 80)

    # ========================================================
    # FIND IMAGES
    # ========================================================

    images = [
        file
        for file in class_folder.iterdir()
        if is_image_file(file)
    ]

    # ========================================================
    # CHECK EMPTY CLASS
    # ========================================================

    if len(images) == 0:

        print(
            "WARNING: No supported images found."
        )

        skipped_classes += 1

        continue

    # ========================================================
    # SHUFFLE
    # ========================================================

    random.shuffle(images)

    total_images_in_class = len(images)

    total_original += total_images_in_class

    # ========================================================
    # CALCULATE SPLIT
    # ========================================================

    train_count = int(
        total_images_in_class
        * TRAIN_RATIO
    )

    validation_count = int(
        total_images_in_class
        * VALIDATION_RATIO
    )

    test_count = (
        total_images_in_class
        - train_count
        - validation_count
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_images = images[
        :train_count
    ]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    # ========================================================
    # DESTINATION DIRECTORIES
    # ========================================================

    train_class_dir = (
        TRAIN_DIR
        / class_name
    )

    validation_class_dir = (
        VALIDATION_DIR
        / class_name
    )

    test_class_dir = (
        TEST_DIR
        / class_name
    )

    train_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # COPY TRAIN
    # ========================================================

    copied_train = copy_images(
        train_images,
        train_class_dir
    )

    # ========================================================
    # COPY VALIDATION
    # ========================================================

    copied_validation = copy_images(
        validation_images,
        validation_class_dir
    )

    # ========================================================
    # COPY TEST
    # ========================================================

    copied_test = copy_images(
        test_images,
        test_class_dir
    )

    # ========================================================
    # UPDATE TOTALS
    # ========================================================

    total_train += copied_train

    total_validation += copied_validation

    total_test += copied_test

    processed_classes += 1

    # ========================================================
    # PRINT CLASS RESULT
    # ========================================================

    print(
        f"\n  Original     : "
        f"{total_images_in_class}"
    )

    print(
        f"  Train        : "
        f"{copied_train}"
    )

    print(
        f"  Validation   : "
        f"{copied_validation}"
    )

    print(
        f"  Test         : "
        f"{copied_test}"
    )

    print(
        f"  Total copied : "
        f"{copied_train + copied_validation + copied_test}"
    )


# ============================================================
# 12. FINAL TOTALS
# ============================================================

total_generated = (
    total_train
    + total_validation
    + total_test
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 80)
print("              DATASET SPLIT COMPLETED")
print("=" * 80)


print(
    f"\nClasses found       : "
    f"{len(class_directories)}"
)

print(
    f"Classes processed   : "
    f"{processed_classes}"
)

print(
    f"Classes skipped     : "
    f"{skipped_classes}"
)

print(
    f"\nOriginal images     : "
    f"{total_original}"
)

print(
    f"Training images     : "
    f"{total_train}"
)

print(
    f"Validation images   : "
    f"{total_validation}"
)

print(
    f"Testing images      : "
    f"{total_test}"
)

print(
    f"Generated images    : "
    f"{total_generated}"
)


# ============================================================
# 14. VERIFY TOTALS
# ============================================================

print("\n" + "=" * 80)
print("DATASET VERIFICATION")
print("=" * 80)


if total_original == total_generated:

    print(
        "\n✓ Image count verification PASSED"
    )

    print(
        "✓ No images were lost during splitting."
    )

else:

    print(
        "\nWARNING: Image count mismatch!"
    )

    print(
        f"Original : {total_original}"
    )

    print(
        f"Generated: {total_generated}"
    )


# ============================================================
# 15. VERIFY APPLE CLASS
# ============================================================

APPLE_CLASS = "Apple___Apple_scab"

apple_source = (
    SOURCE_DIR
    / APPLE_CLASS
)

apple_train = (
    TRAIN_DIR
    / APPLE_CLASS
)

apple_validation = (
    VALIDATION_DIR
    / APPLE_CLASS
)

apple_test = (
    TEST_DIR
    / APPLE_CLASS
)


print("\n" + "=" * 80)
print("APPLE CLASS VERIFICATION")
print("=" * 80)


if apple_source.exists():

    source_count = len(
        [
            file
            for file in apple_source.iterdir()
            if is_image_file(file)
        ]
    )

    print(
        f"\nOriginal Apple___Apple_scab images: "
        f"{source_count}"
    )


else:

    print(
        "\nWARNING: Apple___Apple_scab "
        "does not exist in PlantVillage."
    )


if apple_train.exists():

    apple_train_count = len(
        [
            file
            for file in apple_train.iterdir()
            if is_image_file(file)
        ]
    )

else:

    apple_train_count = 0


if apple_validation.exists():

    apple_validation_count = len(
        [
            file
            for file in apple_validation.iterdir()
            if is_image_file(file)
        ]
    )

else:

    apple_validation_count = 0


if apple_test.exists():

    apple_test_count = len(
        [
            file
            for file in apple_test.iterdir()
            if is_image_file(file)
        ]
    )

else:

    apple_test_count = 0


print(
    f"\nApple___Apple_scab"
)

print(
    f"  Train       : "
    f"{apple_train_count}"
)

print(
    f"  Validation  : "
    f"{apple_validation_count}"
)

print(
    f"  Test        : "
    f"{apple_test_count}"
)

print(
    f"  Total       : "
    f"{apple_train_count + apple_validation_count + apple_test_count}"
)


# ============================================================
# 16. VERIFY ALL GENERATED DIRECTORIES
# ============================================================

print("\n" + "=" * 80)
print("GENERATED DATASET PATHS")
print("=" * 80)


print("\nTrain:")
print(TRAIN_DIR)

print("\nValidation:")
print(VALIDATION_DIR)

print("\nTest:")
print(TEST_DIR)


# ============================================================
# 17. EXPECTED CROP GROUPS
# ============================================================

print("\n" + "=" * 80)
print("CROP GROUPS")
print("=" * 80)


print("\nThe PlantVillage dataset contains classes from:")

print("  ✓ Apple")
print("  ✓ Cherry")
print("  ✓ Corn")
print("  ✓ Grape")
print("  ✓ Orange")
print("  ✓ Peach")
print("  ✓ Pepper")
print("  ✓ Potato")
print("  ✓ Raspberry")
print("  ✓ Soybean")
print("  ✓ Strawberry")
print("  ✓ Tomato")


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("        AgriSpikeNet DATASET PIPELINE READY")
print("=" * 80)


print(
    "\nYou can now use:"
)

print(
    "\n  dataset/train"
)

print(
    "  dataset/validation"
)

print(
    "  dataset/test"
)


print(
    "\nThese directories are ready for "
    "AgriSpikeNet DataLoader and SNN training."
)


print("\n" + "=" * 80)
print("                    DONE")
print("=" * 80)
