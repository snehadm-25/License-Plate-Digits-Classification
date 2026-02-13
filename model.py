import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitClassifier(nn.Module):
    def __init__(self, num_classes=35):
        super(DigitClassifier, self).__init__()
        # Input: 1x32x32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # After 3 MaxPool layers (32 -> 16 -> 8 -> 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 32x32x32
        x = self.pool(F.relu(self.conv2(x))) # 64x16x16
        x = self.pool(F.relu(self.conv3(x))) # 128x8x8
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DigitClassifier()
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
