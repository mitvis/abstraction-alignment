"""Trains model on CIFAR-100 and writes checkpoint to disk."""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import util.cifar.cifar_util as cifar_util


def train(architecture, batch_size, epochs, dataset_directory, model_directory, data_augmentation, seed=None):
    """Train a PyTorch model to classify CIFAR-100 images.."""
    if seed is not None:
        print(f'Setting seed: {seed}.')
        torch.manual_seed(seed)
        if "cuda" in device:
            torch.cuda.manual_seed(seed)
    
    # Load dataset
    train_loader, test_loader = cifar_util.load_dataset(dataset_directory, data_augmentation, batch_size)
    print(f'Loaded CIFAR-100: {len(train_loader.dataset)} train and {len(test_loader.dataset)} test instances.')
    
    # Load model
    cudnn.benchmark = True # Should make training should go faster for large models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_directory, f'{architecture}') 
    model = cifar_util.load_model(architecture)
    model.to(device)
    
    #Set training parameters
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                            momentum=0.9, nesterov=True,
                            weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # Train model
    print(f'Training {architecture} for {epochs} epochs.')
    for epoch in range(epochs):
        train_epoch(epoch, model, train_loader, test_loader, criterion,
                   optimizer, scheduler)

    # Save model checkpoint.
    checkpoint_path = cifar_util.get_checkpoint_path(model_path)
    cifar_util.create_directory(os.path.dirname(checkpoint_path))
    torch.save(model.state_dict(), checkpoint_path)
    
    return model


def test(model, test_loader):
    """Tests the model on the test_loader data and return accuracy."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    model.train()
    return val_acc


def train_epoch(epoch, model, train_loader, test_loader, criterion, 
                optimizer, scheduler):
    """Trains the model for one epoch on the data in the train_loader."""
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_avg = 0.
    correct = 0.
    class_correct = 0
    total = 0.
    
    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        
        images = images.to(device)
        labels = labels.to(device)

        model.zero_grad()
        output = model(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()

        # Calculate running average of accuracy
        pred = torch.max(output.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total
        
        progress_bar.set_postfix(
            loss_fn='%.3f' % (loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
    
    test_acc = test(model, test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step()
