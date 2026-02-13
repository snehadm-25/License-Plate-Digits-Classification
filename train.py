import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import DigitClassifier
import os
from tqdm import tqdm

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Using device: {device}")
    
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size)
    num_classes = len(classes)
    
    model = DigitClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100 * correct / total})
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with acc: {val_acc:.2f}%")
            
    print("Training complete.")
    return model, classes

if __name__ == "__main__":
    data_path = r"c:\License-Plate-Digits-Classification\archive (3)\CNN letter Dataset"
    train_model(data_path, num_epochs=5) # Start with 5 epochs for testing
