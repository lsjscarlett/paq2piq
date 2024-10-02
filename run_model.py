import torch
from paq2piq.model import RoIPoolModel
from paq2piq.inference_model import InferenceModel
from PIL import Image
from pathlib import Path
import os

def process_image(image_path, inference_model):
    """Process and predict for a single image."""
    image = Image.open(image_path)
    output = inference_model.predict_from_pil_image(image)
    print(f"Results for {image_path.name}: {output}")

def main(input_path):
    # Initialize the model
    model = RoIPoolModel()
    inference_model = InferenceModel(model, Path('paq2piq/RoIPoolModel-fit.10.bs.120.pth'))

    input_path = Path(input_path)

    if input_path.is_file():
        # If it's a single image file, process it
        process_image(input_path, inference_model)
    elif input_path.is_dir():
        # If it's a directory, iterate through the images
        for img_file in os.listdir(input_path):
            img_path = input_path / img_file
            if img_file.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
                process_image(img_path, inference_model)
    else:
        print(f"{input_path} is not a valid file or directory.")

if __name__ == "__main__":
    input_path = 'images/photos_sherry/Restaurant'
    main(input_path)
