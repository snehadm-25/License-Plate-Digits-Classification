# License Plate Digits Classification

This project implements an end-to-end CNN-based classifier for license plate digits and letters (0-9 and A-Z).

## Project Structure

- `dataset.py`: Data loading and preprocessing pipeline.
- `model.py`: CNN architecture definitions.
- `train.py`: Training script with validation logic.
- `predict.py`: Inference script for single image classification.
- `best_model.pth`: The trained model weights (approx. 96% accuracy).

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to use

### 1. Training
To train the model from scratch:
```bash
python train.py
```

### 2. Inference
To classify a single digit/letter image, run the prediction script and provide the path to the image as an argument:
```bash
python predict.py path/to/your/image.jpg
```
Example:
```bash
python predict.py "archive (3)\CNN letter Dataset\A\aug18231_0.jpg"
```

## Dataset
The project uses the "CNN letter Dataset" which contains images of digits and uppercase letters organized into folders by class.
