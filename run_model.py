
import os
from pathlib import Path
from PIL import Image
import pillow_heif
from paq2piq.model import RoIPoolModel
from paq2piq.inference_model import InferenceModel

# Register HEIC support with Pillow
pillow_heif.register_heif_opener()

def process_image(image_path, inference_model):
    """Process and predict for a single image."""
    image = Image.open(image_path)
    output = inference_model.predict_from_pil_image(image)
    return output


def main(input_path):
    # Initialize the model
    model = RoIPoolModel()
    inference_model = InferenceModel(model, Path('RoIPoolModel-fit.10.bs.120.pth'))

    input_path = Path(input_path)

    if input_path.is_file():
        # Process single image file
        process_image(input_path, inference_model)

    elif input_path.is_dir():
        # Process directory of images (including case-insensitive .HEIC files)
        img_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))])
        for img_file in img_files:
            img_path = input_path / img_file
            try:
                output = process_image(img_path, inference_model)
                print(f"Results for {img_file}: {output}")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    else:
        print(f"{input_path} is not a valid file or directory.")

if __name__ == "__main__":
    input_path = 'images/photos_sherry/Salon'
    main(input_path)
