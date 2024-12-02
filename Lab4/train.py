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
        # plt.ion()

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

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train(num_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, scaler):
    print('training started at {}'.format(datetime.datetime.now()))
    accumulation_steps = 4

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}, {}".format(epoch, num_epochs, datetime.datetime.now()))
        model.train()
        total_train_loss = 0.0

        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.squeeze(1).to(device, dtype=torch.long)
            # Remap background class (0) to ignore_index (255)
            targets[targets == 0] = 255
            #print("Targets unique values:", torch.unique(targets))  # Add this line
            # forward pass
            with autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
            # backwards pass
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item()

        train_loss = [total_train_loss/len(train_loader)]
        train_losses.append(train_loss)

        # validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.squeeze(1).to(device, dtype=torch.long)

                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_val_loss += loss.item()

        scheduler.step(total_val_loss)
        val_loss = [total_val_loss / len(val_loader)]
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")

        # Clear cache to avoid memory overflow
        torch.cuda.empty_cache()

        torch.save(model.state_dict(), f"models/{filename}.pth")
        print("Model saved.")
        # live_plot.save("unet_curve.png")

        plt.ioff()
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        print(f"saving loss/{filename}_loss.png")
        plt.savefig(f"loss/{filename}_loss.png")


# Main training loop
def main():
    batch_size = 32  # Reduced to fit GTX 1650 VRAM
    num_classes = 21
    num_epochs = 30
    learning_rate = 1e-3
    global filename 
    filename = 'student'

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-f', metavar='filename', type=int, help='the name of the .pth file to save')
    args = argParser.parse_args()

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("loss"):
        os.makedirs("loss")

    if args.e != None:
        num_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.f != None:
        filename = args.f

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    model = student(num_classes=21).to(device)

    # transforms
    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),            # Add random rotations
    transforms.RandomHorizontalFlip(),        # Add horizontal flips
    transforms.ColorJitter(                   # Adjust brightness, contrast, etc.
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
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
    train_dataset = VOCSegmentation(root='./Lab4/data', year='2012',
        image_set='train',
        download=True,
        transform=train_transform,
        target_transform=target_transform
    )
    val_dataset = VOCSegmentation(
        root='./Lab4/data',
        year='2012',
        image_set='val',
        download=True,
        transform=val_transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define the loss function using the computed class weights
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()

    live_plot = LivePlot()
    val_losses = []

    print('\t\tn epochs = ', num_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tlearning rate = ', learning_rate)
    # print('\t\tweight decay = ', w)

    train(
        num_epochs=num_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        scaler=scaler,
        device=device)


if __name__ == "__main__":
    main()
