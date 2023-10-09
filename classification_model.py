import torch
import torch.nn as nn
device = torch.device('cuda')
import torch
import torch.nn as nn
import torch
from torch import nn
from torchvision import models

class ClassificationModel(nn.Module):
    def __init__(self, lr=0.01):
        super().__init__()
        
        self.input_size = 224
        self.output_size = 1
        self.vgg = models.vgg19_bn(models.VGG19_BN_Weights.IMAGENET1K_V1)
        self.lr = lr
        self.best_valid_loss = float('inf')
        
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, 1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.vgg.parameters(), self.lr)
        # print(self.vgg)
    
    def forward(self, x):
        x = self.vgg(x)
        x = torch.sigmoid(x)
        return x
        
    def train_model(self, train_loader, valid_loader, num_epochs=100):
        all_loss = []
        for epoch in range(num_epochs):
            self.vgg.train()
            train_loss = 0
            for img, label in enumerate(train_loader):
                img, label = img.to(device), label.to(device)
                self.optimizer.zero_grad()
                pred = self.vgg(img)
                loss = self.loss_fn(pred, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.vgg.eval()
            with torch.no_grad():
                valid_loss = 0
                for img, label in enumerate(valid_loader):
                    img, label = img.to(device), label.to(device)
                    pred = self.vgg(img)
                    loss = self.loss_fn(pred, label)
                    valid_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            avg_valid_loss = valid_loss / len(valid_loader)
            all_loss.append((avg_train_loss, avg_valid_loss))
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
            if avg_valid_loss < self.best_valid_loss:
                self.best_valid_loss = avg_valid_loss
                torch.save(self.vgg.state_dict(), 'best_model_classification.pth')

