import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(data_dir, batch_size=32, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.classes

if __name__ == "__main__":
    data_path = r"c:\License-Plate-Digits-Classification\archive (3)\CNN letter Dataset"
    train_loader, val_loader, classes = get_dataloaders(data_path)
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
