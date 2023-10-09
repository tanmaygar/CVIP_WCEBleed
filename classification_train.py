import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
device = torch.device('cuda')
import torch
import torch.nn as nn
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np

from dataloader import CustomDataset
from classification_model import ClassificationModel    

def train(model, loader, optimizer, loss_fn):
    epoch_loss = 0.0
    
    model.train()
    for x,y, label in loader:
        x = x.to(device)
        # print(x.shape, y.shape)
        y = y.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn):
    epoch_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for x,y, label in loader:
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            y_pred= model(x)
            loss = loss_fn(y_pred, label)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss/len(loader)
    return epoch_loss


if __name__ == "__main__":
    # Define your data transformations
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.RandomRotation(10),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                    ])

    # Create an instance of the CustomDataset
    dataset = CustomDataset(root_dir='/home/ma22resch11003/CVIP/dataset/WCEBleedGen', transform=transform)
    batch_size = 16
    num_epochs = 30
    lr = 1e-4
    checkpoint_path = "checkpoints/checkpoint.pth"
    # Define the desired train-validation split ratio
    train_ratio = 0.8

    # Calculate the number of samples for each split
    num_total_samples = len(dataset)
    num_train_samples = int(train_ratio * num_total_samples)
    num_val_samples = num_total_samples - num_train_samples

    # Split the dataset into train and validation sets
    train_dataset, valid_dataset = random_split(dataset, [num_train_samples, num_val_samples])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=1)

    model = ClassificationModel()
    # model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.BCELoss()
    best_valid_loss = float("inf")
    train_loss_list = []
    valid_loss_list = []

    """Training the model"""
    for epoch in range(num_epochs):
        start_time = time.time()
        # break
        train_loss = train(model, train_loader, optimizer, loss_fn)
        valid_loss = evaluate(model, valid_loader, loss_fn)
        
        end_time = time.time()
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        epoch_time = (end_time - start_time)/60
        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_time}\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.module.state_dict(), checkpoint_path)

        end_time = time.time()
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        epoch_time = (end_time - start_time)/ 60

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_time}\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        print(data_str)
        # print(data_str)
    best_model = ClassificationModel()
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    best_model.eval()
    labels = []
    y_preds = []
    def accuracy(y_true, y_pred):
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total = len(y_true)
        return correct / total

    def recall(y_true, y_pred):
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
        actual_positives = sum(y_true)
        return true_positives / actual_positives if actual_positives != 0 else 0

    def f1_score(y_true, y_pred):
        prec = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1) / sum(y_pred) if sum(y_pred) != 0 else 0
        rec = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1) / sum(y_true) if sum(y_true) != 0 else 0
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0


    with torch.no_grad():
        for x,y, label in valid_loader:
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            y_pred = best_model(x)
            # labels.append(label.cpu().item())
            # y_preds.append(y_pred.cpu().item() >= 0.5)
            for item in label:
                labels.append(item.item())
            for item in y_pred.cpu():
                if item.item() >= 0.5:
                    y_preds.append(1.0)
                else:
                    y_preds.append(0.0)
    
    print(labels, y_preds)
    acc = accuracy(labels, y_preds)
    rec = recall(labels, y_preds)
    f1 = f1_score(labels, y_preds)

    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}")
    print(f"F1-Score: {f1}")
        
    plt.figure()
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(valid_loss_list, label="Validation Loss")
    plt.legend()
    plt.grid()
    plt.plot()
    plt.savefig("plot_losses.png")
