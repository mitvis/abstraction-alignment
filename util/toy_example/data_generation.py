# Synthetic data generation for a toy abstraction alignment setting.

import numpy as np
import random
from typing import Dict, List, Tuple, Union
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticDataGenerator:
    """Generator for synthetic pattern recognition datasets."""
    
    def __init__(self, image_size=4):
        """Initialize the data generator."""
        self.image_size = image_size
        self.colors = {
            'r': np.array([255, 0, 0], dtype=np.uint8),
            'o': np.array([255, 128, 0], dtype=np.uint8),
            'y': np.array([255, 255, 0], dtype=np.uint8),
            'g': np.array([0, 255, 0], dtype=np.uint8),
            'b': np.array([0, 128, 255], dtype=np.uint8),
            'i': np.array([0, 0, 255], dtype=np.uint8),
            'v': np.array([128, 0, 255], dtype=np.uint8)
        }
        
        # Define corner positions
        self.corners = {
            'topleft': (0, 0),
            'topright': (0, self.image_size-1),
            'botleft': (self.image_size-1, 0),
            'botright': (self.image_size-1, self.image_size-1)
        }
        
        self.midpoint = (self.image_size//2, self.image_size//2)
        
        # Initialize column divisions
        self._initialize_columns()

    def _initialize_columns(self):
        """Initialize column divisions for quadrants."""
        half = self.image_size // 2
        self.column_divisions = {
            'topleft': [[(r, c) for r in range(half)] for c in range(half)],
            'topright': [[(r, c+half) for r in range(half)] for c in range(half)],
            'botleft': [[(r+half, c) for r in range(half)] for c in range(half)],
            'botright': [[(r+half, c+half) for r in range(half)] for c in range(half)]
        }
        
        # Flatten lists for easier access
        for quadrant in self.column_divisions:
            self.column_divisions[quadrant] = [
                pos for col in self.column_divisions[quadrant] for pos in col
            ]

    def _random_color(self, exclude):
        """Generate a random color and return a RGB color array."""
        color_choices = ['r', 'g', 'b', 'v']
        for color_name, color_array in self.colors.items():
            if any(np.array_equal(color_array, exc) for exc in exclude):
                if color_name in color_choices:
                    color_choices.remove(color_name)
        return self.colors[np.random.choice(color_choices)]

    def _change_image(self, image, skip_pixels, skip_color):
        """
        Change colors in the image except for specified pixels.
        
        Args:
            image: Input image array
            skip_pixels: List of pixel coordinates to skip
            skip_color: Color to avoid using
        """
        for row_idx in range(image.shape[0]):
            for col_idx in range(image.shape[1]):
                index = (row_idx, col_idx)
                if index not in skip_pixels and np.array_equal(image[index], skip_color):
                    image[index] = self._random_color([skip_color])

    def generate_class_0(self, image):
        """Generate an image following class 0 rules (three corners same color)."""
        skip_pixels = []
        use_top = np.random.random() < 0.5
        
        if use_top:
            color = image[self.corners['topleft']]
            image[self.corners['topright']] = color
            skip_pixels.extend([self.corners['topleft'], self.corners['topright']])
            
            if np.random.random() < 0.5:
                image[self.corners['botleft']] = color
                skip_pixels.append(self.corners['botleft'])
            else:
                image[self.corners['botright']] = color
                skip_pixels.append(self.corners['botright'])
        else:
            color = image[self.corners['botleft']]
            image[self.corners['botright']] = color
            skip_pixels.extend([self.corners['botleft'], self.corners['botright']])
            
            if np.random.random() < 0.5:
                image[self.corners['topright']] = color
                skip_pixels.append(self.corners['topright'])
            else:
                image[self.corners['topleft']] = color
                skip_pixels.append(self.corners['topleft'])
        
        self._change_image(image, skip_pixels, color)
        return image

    def generate_class_1(self, image):
        """Generate an image following class 1 rules (quadrant patterns)."""        
        patterns = [
            (self.corners['botleft'], self.column_divisions['botleft'] + self.column_divisions['topright']),
            (self.corners['topleft'], self.column_divisions['topleft'] + self.column_divisions['topright']),
            (self.corners['topright'], self.column_divisions['topright'] + self.column_divisions['botright']),
            (self.corners['topleft'], self.column_divisions['topleft'] + self.column_divisions['botright']),
            (self.corners['topleft'], self.column_divisions['topleft'] + self.column_divisions['botleft']),
            (self.corners['botleft'], self.column_divisions['botleft'] + self.column_divisions['botright'])
        ]
        
        pattern_index = np.random.choice(len(patterns))        
        corner, skip_pixels = patterns[pattern_index]
        color = image[corner]
        for pixel in skip_pixels:
            image[pixel] = color
        self._change_image(image, skip_pixels, color)
                
        return image

    def generate_class_2(self, image):
        """Generate an image following class 2 rules (column patterns)."""
        
        # Define the patterns: (source_corner, pixels_to_fill)
        patterns = [
            # Pattern 0: Alternating columns starting with first column
            (self.corners['topleft'], 
             [p for col in range(self.image_size) if col % 2 == 0 
              for p in [(r, col) for r in range(self.image_size)]]),
              
            # Pattern 1: First and third columns
            (self.corners['topleft'], 
             [(r, c) for c in range(0, self.image_size, 2) 
              for r in range(self.image_size)]),
              
            # Pattern 2: Left two columns
            (self.corners['topleft'], 
             self.column_divisions['topleft'] + self.column_divisions['botleft']),
             
            # Pattern 3: Middle two columns
            ((0, 1),  # Special case for middle columns
             [(r, c) for r in range(self.image_size) 
              for c in [1, 2]]),
              
            # Pattern 4: Right two columns
            (self.corners['topright'], 
             self.column_divisions['topright'] + self.column_divisions['botright']),
             
            # Pattern 5: Alternating columns starting with second column
            (self.corners['topright'], 
             [(r, c) for c in range(1, self.image_size, 2) 
              for r in range(self.image_size)])
        ]
        
        # Select pattern based on probability
        idx = np.random.choice(len(patterns))
        source_corner, pixels = patterns[idx]
        color = image[source_corner]

        # Special handling for middle columns pattern
        if idx == 3:  # Pattern 3 (middle columns)
            # Ensure corners are different to avoid confusion with class 0
            corner_positions = [self.corners['topleft'], self.corners['topright'],
                             self.corners['botleft'], self.corners['botright']]
            available_colors = []
            used_color = image[source_corner]

            # Find available colors that are different from the middle color
            for c in ['r', 'g', 'b', 'v']:
                if not np.array_equal(self.colors[c], used_color):
                    available_colors.append(self.colors[c])

            # Randomly assign different colors to corners
            random.shuffle(available_colors)
            for i, corner in enumerate(corner_positions):
                image[corner] = available_colors[i % len(available_colors)]

        # Fill the pattern pixels with the selected color
        for pixel in pixels:
            image[pixel] = color

        # Change other pixels to different colors
        self._change_image(image, pixels, color)
                
        return image

    def generate_image(self, label):
        """Generate a synthetic image with specified label."""
        image = np.array([[self._random_color([]) for _ in range(self.image_size)]
                         for __ in range(self.image_size)], dtype=np.uint8)
        
        assert label in set(range(3)), f'Called generate_image with label {label}'
        
        if label == 0:
            return self.generate_class_0(image)
        elif label == 1:
            return self.generate_class_1(image)
        elif label == 2:
            return self.generate_class_2(image)

    def show_examples(self, n_per_class=5):
        """Generate and show n examples of each pattern class."""

        # Create subplot grid
        fig, axes = plt.subplots(3, n_per_class, figsize=(2*n_per_class, 6))
        fig.suptitle('Dataset Examples', fontsize=14)

        # Generate and show images for each class
        for class_label in range(3):
            for i in range(n_per_class):
                image = self.generate_image(class_label)
                axes[class_label, i].imshow(image)
                axes[class_label, i].axis('off')

            # Add class label to first image in row
            axes[class_label, 0].set_title(f'Class {class_label}')

        plt.tight_layout()
        plt.show()
    
    
    
class SyntheticDataset(Dataset):
    """Dataset class for synthetic patterns."""
    
    def __init__(self, num_samples_per_class, shuffle=False):
        """
        Initialize the pattern dataset.
        
        Args:
            num_samples_per_class: Number of samples to generate for each class.
            shuffle: Whether or not to shuffle the dataset
        """
        self.generator = SyntheticDataGenerator()
        
        # Generate labels
        labels = []
        for class_idx in range(3):  # 3 classes
            labels.extend([class_idx] * num_samples_per_class)
        
        # Shuffle labels
        if shuffle:
            np.random.shuffle(labels)
        self.labels = np.array(labels)
        
        # Generate all images
        self.images = np.array([
            self.generator.generate_image(label) 
            for label in self.labels
        ])
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset at idx.
            
        Returns:
            tuple: (image, label)
                image: Flattened and normalized image tensor
                label: Class label
        """
        # Get and preprocess image
        image = self.images[idx]
        image_tensor = torch.FloatTensor(image.ravel()) / 255.0  # Flatten and normalize
        
        return image_tensor, self.labels[idx]

    
def create_dataloaders(
    train_samples_per_class=1500, test_samples_per_class=500, batch_size=32,
    num_workers=4
):
    """
    Create train and test dataloaders.
    
    Args:
        train_samples_per_class: Number of training samples per class
        test_samples_per_class: Number of test samples per class
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = SyntheticDataset(train_samples_per_class, shuffle=True)
    test_dataset = SyntheticDataset(test_samples_per_class, shuffle=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader