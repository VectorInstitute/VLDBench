import os
import argparse
import imageio
import imgaug.augmenters as iaa
from PIL import Image

# Define imgaug augmenters
AUGMENTERS = {
    "blur": iaa.GaussianBlur(sigma=0.5),  # Approximate (3,3) kernel effect
    "noise": iaa.AdditiveGaussianNoise(loc=0, scale=0.1 * 255),
    "scale_0.5": iaa.Resize(0.5),
    "scale_2": iaa.Resize(2.0),
}


def apply_augmentation(input_folder, output_folder, perturbation):
    os.makedirs(output_folder, exist_ok=True)

    perturbations_to_apply = list(AUGMENTERS.keys()) if perturbation == "all" else [perturbation]

    for pert in perturbations_to_apply:
        augmenter = AUGMENTERS[pert]
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for filename in image_files:
            input_path = os.path.join(input_folder, filename)

            # Read image
            image = imageio.imread(input_path)

            # Apply augmentation
            augmented_image = augmenter(image=image)

            # Create new filename with augmentation label
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{pert}{ext}"
            output_path = os.path.join(output_folder, new_filename)

            # Save result
            Image.fromarray(augmented_image).save(output_path)

        print(f"'{pert}' to {len(image_files)} image(s).")

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image perturbations using imgaug.")
    parser.add_argument("--input_folder", required=True, help="Folder with input images.")
    parser.add_argument("--output_folder", required=True, help="Folder to save augmented images.")
    parser.add_argument("--perturbation", default="all",choices=["blur", "noise", "scale_0.5", "scale_2", "all"],
                        help="Type of perturbation to apply (default: all).")
    args = parser.parse_args()

    apply_augmentation(args.input_folder, args.output_folder, args.perturbation)

# To Run particular perturbation:
# python image_perturbations.py --input_folder path/to/input --output_folder path/to/output --perturbation blur
# python image_perturbations.py --input_folder path/to/input --output_folder path/to/output --perturbation noise
# python image_perturbations.py --input_folder path/to/input --output_folder path/to/output --perturbation scale_0.5
# python image_perturbations.py --input_folder path/to/input --output_folder path/to/output --perturbation scale_2

# To Run all perturbations:
# python image_perturbations.py --input_folder path/to/input --output_folder path/to/output --perturbation all