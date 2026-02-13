import torch
from torchvision import transforms
from PIL import Image
from model import DigitClassifier
import os

def predict(image_path, model_path='best_model.pth', class_names=None):
    if class_names is None:
        # Default classes based on dataset exploration
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                       'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 
                       'V', 'W', 'X', 'Y', 'Z']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DigitClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    return class_names[predicted.item()]

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify a license plate digit/letter image.')
    parser.add_argument('image', type=str, nargs='?', help='Path to the image file to classify')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to the trained model file')
    
    args = parser.parse_args()
    
    image_path = args.image
    if not image_path:
        image_path = input("Please enter the path to the image file: ").strip().strip('"').strip("'")
    
    if os.path.exists(args.model):
        if os.path.exists(image_path):
            result = predict(image_path, model_path=args.model)
            print(f"Predicted class: {result}")
        else:
            print(f"Error: Image file '{image_path}' not found.")
    else:
        print(f"Error: Model file '{args.model}' not found. Please train the model first.")
