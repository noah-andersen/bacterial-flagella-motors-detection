# import needed packages
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from torchvision import transforms as v2

def create_image_map(): 
    """
    Create a dictionary with image names as keys and a dictionary with x, y, and motor_visible as values.
    Example: {"0001.png": {"x": 223, "y": 323, "motor_visible": 1}, ...}
    """
    if not os.path.exists('image_name_to_coordinates.json'):
        train_df = pd.read_csv('train.csv')
        class_map = {}
        for i in range(len(train_df)):
            if train_df.iloc[i,0] not in class_map:
                class_map[train_df.iloc[i,0]] = {}
                class_map[train_df.iloc[i,0]]['x'] = int(train_df.iloc[i,1])
                class_map[train_df.iloc[i,0]]['y'] = int(train_df.iloc[i,2])
                class_map[train_df.iloc[i,0]]['motor_visible'] = int(train_df.iloc[i,3])
        with open('image_name_to_coordinates.json', 'w') as f:
            json.dump(class_map, f)
    else:
        with open('image_name_to_coordinates.json', 'r') as f:
            class_map = json.load(f)
    return class_map

def train_test_split(test_ratio:float):
    """
    Create a training and testing split of the images.

    Parameters:
    test_ratio: float
        The ratio of the test set size to the total dataset size.
    """
    images = os.listdir('train')
    np.random.shuffle(images)
    split = int(len(images)*test_ratio)
    test_images = images[:split]
    train_images = images[split:]
    return train_images, test_images

# Create function to show images with object point annotated
def show_sample(num_cols:int, num_rows:int, train=True):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    object_map = create_image_map()
    branch = 'train' if train else 'val'
    plt.suptitle(f'Bacterial Flaggella {branch} Images Sample', fontsize=20, fontweight='bold')
    sample = np.random.choice(os.listdir(branch), num_cols*num_rows)
    for i in range(num_rows):
        for j in range(num_cols):
            image_path = f'train/{sample[i*num_cols+j]}'
            image_name = sample[i*num_cols+j]
            x,y,motor_visible = object_map[image_name]['x'], object_map[image_name]['y'], object_map[image_name]['motor_visible']
            image = PIL.Image.open(image_path)
            axs[i, j].imshow(image, cmap='gray')
            if motor_visible:
                axs[i, j].scatter(x, y, c='r', s=200, marker='x')
            axs[i, j].set_title(sample[i*num_cols+j])
            axs[i, j].axis('off')
    plt.show()

def color_jitter():
    """
    Apply the ColorJitter transformation to a sample image with varying brightness values.
    """
    # Check to see what colorjitter brightenss would do to the image
    random_image = np.random.choice(os.listdir('train'),1)[0]
    image_path = f"train/{random_image}"
    image = PIL.Image.open(image_path)

    # Define brightness values to iterate over
    brightness_values = np.linspace(0.25, 1.75, 10)  # Generate 10 brightness values from 0.5 to 1.5

    # Define ColorJitter transformation with only brightness adjustment
    color_jitter = v2.ColorJitter(brightness=(0, 0))  # We'll set brightness dynamically in the loop
    object_map = create_image_map()
    x,y,motor_visible = object_map[random_image]['x'],object_map[random_image]['y'],object_map[random_image]['motor_visible']

    # Plot the transformed images in a 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("ColorJitter Transformation with Varying Brightness", fontsize=16, y=1.02, fontweight='bold')
    for i, brightness in enumerate(brightness_values):
        # Set brightness value for the ColorJitter transformation
        color_jitter.brightness = (brightness.item(), brightness.item())
        
        # Apply the ColorJitter transformation to the image
        transformed_image = color_jitter(image)
        
        # Plot the transformed image
        row = i // 5
        col = i % 5
        axes[row, col].imshow(transformed_image, cmap='gray')
        if motor_visible:
            axes[row, col].scatter(x, y, c='r', s=200, marker='x')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Brightness: {brightness:.2f}")

    plt.tight_layout()
    plt.show()

def gaussian_blur():
    """
    Apply the GaussianBlur transformation to a sample image with varying brightness values. 
    """
    # Check to see what colorjitter brightenss would do to the image
    random_image = np.random.choice(os.listdir('train'),1)[0]
    image_path = f"train/{random_image}"
    image = PIL.Image.open(image_path)

    # Define brightness values to iterate over
    blur_values = np.linspace(0.2, 2, 10)  # Generate 10 brightness values from 0.5 to 1.5

    # Define ColorJitter transformation with only brightness adjustment
    object_map = create_image_map()
    x,y,motor_visible = object_map[random_image]['x'],object_map[random_image]['y'],object_map[random_image]['motor_visible']

    # Plot the transformed images in a 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Gaussian Blur Transformation with Varying Sigma Values", fontsize=16, y=1.02, fontweight='bold')
    for i, blurs in enumerate(blur_values):
        # Set brightness value for the ColorJitter transformation
        blur = v2.GaussianBlur(kernel_size=3, sigma=blurs)
        
        # Apply the ColorJitter transformation to the image
        transformed_image = blur(image)
        
        # Plot the transformed image
        row = i // 5
        col = i % 5
        axes[row, col].imshow(transformed_image, cmap='gray')
        if motor_visible:
            axes[row, col].scatter(x, y, c='r', s=200, marker='x')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Sigma: {blurs:.2f}")

    plt.tight_layout()
    plt.show()
