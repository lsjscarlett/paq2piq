import os
from pathlib import Path
from PIL import Image
import pillow_heif
from paq2piq.inference_model import InferenceModel
from paq2piq.common import Transform, render_output
from paq2piq.model import RoIPoolModel

# Register HEIC support with Pillow
pillow_heif.register_heif_opener()

def process_image(image_path, inference_model, render=False, output_dir=None):
    """Process and predict for a single image."""
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"[ERROR] Failed to open image {image_path}: {e}")
        return None
    try:
        output = inference_model.predict_from_pil_image(image)
    except Exception as e:
        print(f"[ERROR] Failed to predict image {image_path}: {e}")
        return None

    # Optional rendering of quality maps
    if render and output_dir is not None:
        try:
            rendered_image = render_output(image, output)
            output_image_path = Path(output_dir) / f"{image_path.stem}_rendered.png"
            rendered_image.save(output_image_path)
            print(f"[INFO] Rendered output saved to {output_image_path}")
        except Exception as e:
            print(f"[ERROR] Failed to render image {image_path}: {e}")

    return output  # Ensure that the output is returned

def display_results(image_name, output):
    """Display the prediction results."""
    if output is None:
        print(f"[WARNING] No output to display for {image_name}.\n")
        return

    print(f"--- Results for {image_name} ---")
    print(f"Global Score: {output.get('global_score', 'N/A')}")
    print(f"Normalized Global Score: {output.get('normalized_global_score', 'N/A')}")
    print(f"Average Local Score: {output.get('average_local_score', 'N/A')}")
    print(f"Category: {output.get('category', 'N/A')}")
    print(f"Rescaled Global Score: {output.get('rescaled_global_score', 'N/A')}")
    print(f"Rescaled Normalized Global Score: {output.get('rescaled_normalized_global_score', 'N/A')}")
    print(f"Rescaled Average Local Score: {output.get('rescaled_average_local_score', 'N/A')}")
    print(f"Rescaled Category: {output.get('rescaled_category', 'N/A')}")
    print("-----------------------------------\n")

def main(input_path, render=False, output_dir=None):
    # Initialize the model
    model = RoIPoolModel()
    model_state_path = Path(__file__).parent / 'RoIPoolModel-fit.10.bs.120.pth'

    # Debugging: Check if model state file exists
    print(f"Model State Path: {model_state_path}")
    print(f"Model State Exists: {model_state_path.exists()}")

    if not model_state_path.exists():
        print(f"[ERROR] Model state file '{model_state_path}' does not exist.")
        return

    inference_model = InferenceModel(model, model_state_path)

    input_path = Path(input_path)

    # Debugging statements
    print(f"Input Path: {input_path}")
    print(f"Exists: {input_path.exists()}")
    print(f"Is File: {input_path.is_file()}")
    print(f"Is Directory: {input_path.is_dir()}")

    # Prepare output directory for rendered images if rendering is enabled
    if render:
        if output_dir is not None:
            output_dir = Path(output_dir)
        else:
            output_dir = Path('rendered_outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output directory for rendered images: {output_dir}")

    # Check if the input path is a file
    if input_path.is_file():
        print(f"[INFO] Processing single image file: {input_path}")
        # Process single image file
        output = process_image(input_path, inference_model, render, output_dir)
        display_results(input_path.name, output)

    # Check if the input path is a directory
    elif input_path.is_dir():
        print(f"[INFO] Processing directory of images: {input_path}")
        # Process directory of images (including case-insensitive .HEIC files)
        img_files = sorted([
            f for f in os.listdir(input_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))
        ])
        if not img_files:
            print(f"[WARNING] No images found in directory {input_path}")
            return

        for img_file in img_files:
            img_path = input_path / img_file
            print(f"[INFO] Processing image: {img_file}")
            output = process_image(img_path, inference_model, render, output_dir)
            display_results(img_file, output)

    else:
        print(f"[ERROR] {input_path} is not a valid file or directory.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PAQ2PIQ Image Quality Assessment")
    parser.add_argument('input_path', type=str, help="Path to an image file or directory of images.")
    parser.add_argument('--render', action='store_true', help="Render quality maps on images.")
    parser.add_argument('--output_dir', type=str, default='rendered_outputs', help="Directory to save rendered images.")

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        render=args.render,
        output_dir=args.output_dir if args.render else None
    )
