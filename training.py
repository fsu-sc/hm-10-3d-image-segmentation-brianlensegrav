import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dice_loss import dice_loss  # Implement separately

def train_model(model, dataloaders, num_epochs=50, lr=1e-4, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, masks in dataloaders['train']:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks in dataloaders['val']:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += dice_loss(outputs, masks).item()

        writer.add_scalars('Loss', {'train': total_loss, 'val': val_loss}, epoch)
        print(f"Epoch {epoch}: Train Loss={total_loss:.4f}, Val Loss={val_loss:.4f}")
