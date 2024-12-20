import torch
import numpy as np
from pathlib import Path
from PIL import Image
from math import exp
from .common import Transform, render_output  # Ensure 'common.py' contains these
from .model import RoIPoolModel  # Ensure 'model.py' contains 'RoIPoolModel'

class InferenceModel:
    """
    InferenceModel handles the prediction and rescaling of image quality scores using the PAQ2PIQ model.
    """
    blk_size = (20, 20)  # Define the block size for RoIPool
    categories = ('Poor', 'Fair', 'Good', 'Excellent')

    def __init__(self, model: RoIPoolModel, path_to_model_state: Path = None):
        """
        Initialize the InferenceModel with the specified model and load the state if provided.

        Args:
            model (RoIPoolModel): The PAQ2PIQ RoIPoolModel instance.
            path_to_model_state (Path, optional): Path to the model state file. Defaults to None.
        """
        self.transform = Transform().val_transform  # Define the transformation pipeline
        self.model = model.to(self.device)  # Move model to the appropriate device (CPU or GPU)
        self.model.eval()  # Set model to evaluation mode

        if path_to_model_state is not None:
            if not path_to_model_state.exists():
                raise FileNotFoundError(f"Model state file '{path_to_model_state}' does not exist.")
            # Load the model state dict
            model_state = torch.load(path_to_model_state, map_location=self.device)
            self.model.load_state_dict(model_state["model"])

        # Rescaling parameters based on training data statistics
        self.x_mean = 72.59696108881171
        self.std_left = 7.798274017370107
        self.std_right = 4.118047289170692

    @property
    def device(self):
        """Determine the device to run the model on."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rescale_score(self, global_score: float) -> float:
        """
        Exponential rescaling of the global score.

        Args:
            global_score (float): The original global score from the model.

        Returns:
            float: The rescaled global score.
        """
        factor = 3.0
        x = self.x_mean + factor * (global_score - self.x_mean)
        return max(0, min(x, 100))


    def determine_rescaled_category(self, rescaled_scores: list) -> str:
        """
        Determine the rescaled category based on average rescaled scores.

        Args:
            rescaled_scores (list): List of rescaled local scores.

        Returns:
            str: The rescaled category.
        """
        average_rescaled = np.mean(rescaled_scores)
        category_index = int(average_rescaled // 20)
        category_index = min(category_index, len(self.categories) - 1)  # Ensure index is within bounds
        return self.categories[category_index]

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        # Apply transformations and add batch dimension
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)

        # Prepare RoI Pooling based on image dimensions
        self.model.input_block_rois(self.blk_size, [transformed_image.shape[-2], transformed_image.shape[-1]],
                                    device=self.device)

        # Forward pass through the model
        output = self.model(transformed_image).cpu().numpy()[0]

        # Extract scores
        global_score = output[0]
        local_scores = np.reshape(output[1:], self.blk_size)
        average_local_score = np.mean(local_scores)

        # Rescale scores
        normalized_global_score = self.rescale_score(global_score)  # Normalized global score
        rescaled_global_score = self.rescale_score(global_score)  # Rescale global score
        rescaled_normalized_global_score = self.rescale_score(
            normalized_global_score)  # Rescale normalized global score
        rescaled_average_local_score = self.rescale_score(average_local_score)  # Rescale average local score

        # Define original category based on normalized global score (4 buckets)
        if normalized_global_score <= 40:
            category = self.categories[0]  # 'Poor'
        elif 40 < normalized_global_score <= 60:
            category = self.categories[1]  # 'Fair'
        elif 60 < normalized_global_score <= 80:
            category = self.categories[2]  # 'Good'
        else:
            category = self.categories[3]  # 'Excellent'

        # Define rescaled category based on rescaled normalized global score (4 buckets)
        if rescaled_normalized_global_score <= 40:
            rescaled_category = self.categories[0]
        elif 40 < rescaled_normalized_global_score <= 60:
            rescaled_category = self.categories[1]
        elif 60 < rescaled_normalized_global_score <= 80:
            rescaled_category = self.categories[2]
        else:
            rescaled_category = self.categories[3]

        # Compile results
        results = {
            "global_score": float(global_score),
            "normalized_global_score": float(normalized_global_score),
            "average_local_score": float(average_local_score),
            "category": category,  # Based on normalized global score
            "rescaled_global_score": float(rescaled_global_score),
            "rescaled_normalized_global_score": float(rescaled_normalized_global_score),
            "rescaled_average_local_score": float(rescaled_average_local_score),
            "rescaled_category": rescaled_category,  # Based on rescaled normalized global score
        }

        return results

    def predict_from_pil_image(self, image: Image.Image) -> dict:
        """
        Convenience method to predict from a PIL Image.

        Args:
            image (PIL.Image.Image): The input PIL Image.

        Returns:
            dict: Prediction results.
        """
        image = image.convert("RGB")  # Ensure image is in RGB format
        return self.predict(image)