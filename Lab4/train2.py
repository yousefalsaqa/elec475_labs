#########################################################################################################
#
#   ELEC 475 - Project
#   Fall 20234
#

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import datetime
import argparse

from model import *

import pdb

#   some default parameters, which can be overwritten by command line arguments
save_file = 'weights.pth'
n_epochs = 30
batch_size = 256
learning_rate = 1e-3
weight_decay = 1e-5
early_stopping_patience = 5  # Stop training if val_loss doesn't improve for 5 epochs

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, save_file=None):
    print('training started at '.format(datetime.datetime.now()))

    losses_train = []
    losses_val = []

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, n_epochs+1):

        print('epoch ', epoch)

        # training phase
        model.train()
        loss_train = 0.0
        for step, (imgs, targets) in enumerate(train_loader):
            pdb.set_trace()
            imgs = imgs.to(device=device)
            targets = targets.squeeze(1).to(device=device, dtype=torch.long)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            # print(loss_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]

        # Validation phase
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for step, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.to(device=device)
                targets = targets.squeeze(1).to(device=device, dtype=torch.long)

                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                loss_val += loss.item()

        scheduler.step(loss_val)
        losses_val += [loss_val/len(val_loader)]


        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))
        print('{} Epoch {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, loss_val/len(val_loader)))

        if save_file != None:
            torch.save(model.state_dict(), f"models/student_{batch_size}_{learning_rate}_{weight_decay}.pth")

        # if plot_file != None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        print('saving ', f"loss/student_{batch_size}_{learning_rate}_{weight_decay}.png")
        plt.savefig(f"loss/student_{batch_size}_{learning_rate}_{weight_decay}.png")

        # Early stopping
        if loss_val/len(val_loader) < best_val_loss:
            best_val_loss = loss_val/len(val_loader)
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter} epochs")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered. Stopping training...")
                break


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():

    global save_file, n_epochs, batch_size, learning_rate, weight_decay

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-l', metavar='learning rate', type=float, help='learning rate')
    argParser.add_argument('-w', metavar='weight decay', type=float, help='weight decay')


    args = argParser.parse_args()

    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.l != None:
        learning_rate = args.l
    if args.w != None:
        weight_decay = args.w

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    model = student(in_channels=3, num_classes=21)
    model.to(device)
    model.apply(init_weights)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

    # VOC_COLORMAP = np.array([
    #     [0, 0, 0],
    #     [128, 0, 0],
    #     [0, 128, 0],
    #     [128, 128, 0],
    #     [0, 0, 128],
    #     [128, 0, 128],
    #     [0, 128, 128],
    #     [128, 128, 128],
    #     [64, 0, 0],
    #     [192, 0, 0],
    #     [64, 128, 0],
    #     [192, 128, 0],
    #     [64, 0, 128],
    #     [192, 0, 128],
    #     [64, 128, 128],
    #     [192, 128, 128],
    #     [0, 64, 0],
    #     [128, 64, 0],
    #     [0, 192, 0],
    #     [128, 192, 0],
    #     [0, 64, 128],
    # ], dtype=np.uint8)

    # # Create a LUT to map RGB values to class indices
    # VOC_LUT = {tuple(color): idx for idx, color in enumerate(VOC_COLORMAP)}

    # # Function to convert RGB mask to class index mask
    # def rgb_to_class_index(mask):
    #     # Ensure mask is in the right format
    #     mask = np.array(mask, dtype=np.uint8)
    #     class_index_mask = np.zeros(mask.shape[:2], dtype=np.int64)

    #     # Map each pixel to its corresponding class index
    #     for rgb, class_idx in VOC_LUT.items():
    #         class_index_mask[(mask == rgb).all(axis=-1)] = class_idx

    #     return class_index_mask

    # class FixMask:
    #     def __call__(self, img):
    #         class_index_mask = rgb_to_class_index(img)
    #         # Resize using nearest neighbor interpolation
    #         resize_transform = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
    #         class_index_mask = Image.fromarray(class_index_mask.astype(np.uint8))
    #         resized_mask = resize_transform(class_index_mask)
        
    #         # Convert back to tensor
    #         return torch.tensor(np.array(resized_mask), dtype=torch.long)

    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        # FixMask(),
        transforms.PILToTensor()
    ])

    train_set = VOCSegmentation(root='./data', 
                                year='2012',
                                image_set='train',
                                download=True,
                                transform=train_transform,
                                target_transform=target_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_set = VOCSegmentation(root='./data', 
                            year='2012',
                            image_set='trainval',
                            download=True,
                            transform=val_transform,
                            target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    weights = 1/(np.array([1, 108, 94, 137, 124, 195, 121, 209, 154, 303, 152, 86, 149, 100, 101, 868, 151, 155, 103, 96, 101]))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0, reduction='mean')
    
    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tlearning rate = ', learning_rate)
    print('\t\tweight decay = ', weight_decay)
    
    train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        save_file=save_file)


###################################################################

if __name__ == '__main__':
    main()

