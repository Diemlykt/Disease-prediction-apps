import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        x = self.pool(x)
        
        return x
    
class LinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.fc = nn.Linear(in_channel, out_channel)
        self.batch_norm = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convblock1 = ConvBlock(1, 32, 64)
        self.convblock2 = ConvBlock(64, 128, 128)
        self.convblock3 = ConvBlock(128, 256, 256)
        self.convblock4 = ConvBlock(256, 512, 512)
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.linearblock1 = LinearBlock(512 * 15 * 15, 1024)
        self.linearblock2 = LinearBlock(1024, 512)
        self.linearblock3 = LinearBlock(512, 16)
        
        self.linearblock4 = LinearBlock(16 + 2, 4)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, img, mean, std):
        x = self.convblock1(img)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        
        x = self.flatten(x)
        
        x = self.linearblock1(x)
        x = self.linearblock2(x)
        x = self.linearblock3(x)
        
        x = torch.concat([x, mean.unsqueeze(1), std.unsqueeze(1)], dim=-1)
        x = self.linearblock4(x)
        x = self.softmax(x)
        
        return x