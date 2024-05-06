"""Util for datasets and transforms."""

import numpy as np
import resnet
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms, datasets
from matplotlib import pyplot as plt


CIFAR_MEAN = np.array([125.3, 123.0, 113.9])
CIFAR_STD = np.array([63.0, 62.1, 66.7])

CIFAR_NORMALIZE = transforms.Normalize(mean=[x / 255.0 for x in CIFAR_MEAN],
                                       std=[x / 255.0 for x in CIFAR_STD])
CIFAR_RANDOM_CROP = transforms.RandomCrop(32, padding=4)
CIFAR_RANDOM_FLIP = transforms.RandomHorizontalFlip()

NUM_CLASSES = 100


def load_model(architecture):
    if architecture in ['resnet%i' %(i) for i in [20, 32, 44, 56, 110, 1202]]:
        model = resnet.__dict__[architecture](NUM_CLASSES)
    
    else:
        model = models.__dict__[architecture](pretrained=False)
        if 'resnet' in architecture:
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif 'alexnet' == architecture or 'vgg' in architecture:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        elif 'squeezenet' in architecture:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        elif 'densenet' in archtecture:
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        elif 'inception' in architecture:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def load_dataset(dataset_directory, data_augmentation, batch_size):
    train_transform = cifar_train_transform(data_augmentation)
    test_transform = cifar_test_transform()

    train_dataset = datasets.CIFAR100(root=dataset_directory,
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root=dataset_directory,
                                     train=False,
                                     transform=test_transform,
                                     download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)
    return train_loader, test_loader


def cifar_train_transform(data_augmentation):
    train_transform = transforms.Compose([])
    if data_augmentation:
        train_transform.transforms.append(CIFAR_RANDOM_CROP)
        train_transform.transforms.append(CIFAR_RANDOM_FLIP)
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(CIFAR_NORMALIZE)
    return train_transform


def cifar_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        CIFAR_NORMALIZE,
    ])
    return test_transform


def unnorm_cifar_image(x):
    # Unnormalize image by undoing mean/std normalization.
    # Image is kept in the range [0, 1].
    x_unnorm = x * (CIFAR_STD / 255.0) + (CIFAR_MEAN / 255.0)
    x_unnorm = np.clip(x_unnorm, 0, 1)
    return x_unnorm


def transpose_channel_last(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if len(x.shape) == 4:  # Includes batch dimension.
        order = (0, 2, 3, 1)
    else:
        order = (1, 2, 0)
    return np.transpose(x, order)


def plot_cifar_image(x, unnorm=False):
    if x.shape[0] == 3:  # Transpose channel last.
        x = transpose_channel_last(x)
    if unnorm:
        x = unnorm_cifar_image(x)
    plt.imshow(x)
    plt.axis('off')
