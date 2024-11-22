import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.amp import autocast, GradScaler

from model import student
import os
import matplotlib.pyplot as plt
import argparse
import datetime

# Enable cuDNN benchmark mode for better performance
torch.backends.cudnn.benchmark = True

# Training function with gradient accumulation and mixed precision
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.squeeze(1).to(device, dtype=torch.long)

        with autocast(device_type="cuda"):
            outputs = model(imgs)
            loss = loss_fn(outputs, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.squeeze(1).to(device, dtype=torch.long)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Dynamic learning rate adjustment
def adjust_learning_rate(optimizer, val_losses, threshold=0.01, reduce_factor=0.7, increase_factor=1.3, min_lr=1e-5, max_lr=1e-3):
    if len(val_losses) > 3:
        recent_losses = val_losses[-3:]
        improvement = recent_losses[-1] - min(recent_losses)

        if improvement > -threshold:
            for param_group in optimizer.param_groups:
                new_lr = max(param_group['lr'] * reduce_factor, min_lr)
                param_group['lr'] = new_lr
                print(f"Learning rate reduced to: {new_lr:.6f}")
        elif improvement < -2 * threshold:
            for param_group in optimizer.param_groups:
                new_lr = min(param_group['lr'] * increase_factor, max_lr)
                param_group['lr'] = new_lr
                print(f"Learning rate increased to: {new_lr:.6f}")

# Live plot class
class LivePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

        self.ax.set_title("Training and Validation Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.train_line, = self.ax.plot([], [], label="Train Loss", marker="o")
        self.val_line, = self.ax.plot([], [], label="Validation Loss", marker="x")
        self.ax.legend()
        plt.ion()

    def update(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filename):
        plt.ioff()
        plt.savefig(filename)
        print(f"Loss curve saved as '{filename}'.")

# Main training loop
def main():
    batch_size = 64  # Reduced to fit GTX 1650 VRAM
    num_classes = 21
    num_epochs = 35
    learning_rate = 1e-3

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    args = argParser.parse_args()

    if args.e != None:
        num_epochs = args.e
    if args.b != None:
        batch_size = args.b

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    # data loaders
    train_dataset = VOCSegmentation(root='./data', year='2012',
        image_set='train',
        download=True,
        transform=train_transform,
        target_transform=target_transform
    )
    val_dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=True,
        transform=val_transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = student(num_classes=num_classes).to(device)
    # print(summary(model=model, input_size=(1, 3, 256, 256)))
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()

    live_plot = LivePlot()
    val_losses = []

    print('\t\tn epochs = ', num_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tlearning rate = ', learning_rate)
    # print('\t\tweight decay = ', w)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        live_plot.update(epoch, train_loss, val_loss)

        adjust_learning_rate(optimizer, val_losses)

        # Clear cache to avoid memory overflow
        torch.cuda.empty_cache()

    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(model.state_dict(), "./models/student.pth")
    print("Model saved.")
    live_plot.save("student_curve.png")


if __name__ == "__main__":
    main()
