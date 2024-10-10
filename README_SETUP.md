# PaQ2PiQ: Image Quality Prediction Model

## Overview
PaQ2PiQ is a deep learning-based image quality assessment model designed to predict perceptual image quality using a pre-trained ResNet-18 backbone. 

Original Post: https://github.com/baidut/paq2piq
## Installation
Clone the project from Git
```bash
git clone https://github.com/lsjscarlett/paq2piq
```
Install the required packages
```bash
pip install -r requirements.txt
```
Download the pre-trained package from  [GitHub Repository]([GitHub Repository](https://github.com/your-repo) to the repository
) 

## Usage
### For single image
``` bash
python run_model.py --image_path /path/to/image.jpg --model_path /path/to/pretrained_model.pth
```
### For batch images
``` bash
python run_model.py --folder_path /path/to/folder --model_path /path/to/pretrained_model.pth
```

## Output
The output will display:

Global Score: The overall quality score for each image.

Local Scores: A heatmap of quality scores over different regions of the image.

Example:
```json
{
  "global_score": 76.8,
  "normalized_global_score": 79.4,
  "local_scores": [...],
  "category": "Good"
}
