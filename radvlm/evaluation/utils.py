import torch
from PIL import Image
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
import numpy as np
import math
import os

from radvlm.evaluation.compute_metrics_tasks import extract_bounding_boxes

import os
import sys 


def display_image_with_bboxes(ax, img_path, predicted_bboxes, actual_bboxes, label=None):
    # Load the image
    img = Image.open(img_path)
    img_np = np.array(img)
    # Get image dimensions
    img_width, img_height = img.size
    # Display the image
    ax.imshow(img_np, cmap="grey")
    # Add actual bounding boxes in green
    for bbox in actual_bboxes:
        # Convert bbox coordinates from relative to absolute
        x_min = bbox[0] * img_width
        y_min = bbox[1] * img_height
        width = (bbox[2] - bbox[0]) * img_width
        height = (bbox[3] - bbox[1]) * img_height
        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=4, edgecolor='lightgreen', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    # Add predicted bounding boxes in red
    for bbox in predicted_bboxes:
        # Convert bbox coordinates from relative to absolute
        x_min = bbox[0] * img_width
        y_min = bbox[1] * img_height
        width = (bbox[2] - bbox[0]) * img_width
        height = (bbox[3] - bbox[1]) * img_height
        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=4, edgecolor='red', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    # Set the title of the subplot
    if label is not None:   
        ax.set_title(label)


def plot_images_with_Bbox(output, results_dir, num_samples=10):
    # Ensure we don't exceed the available number of outputs
    num_samples = min(num_samples, len(output))
    
    # Get the last `num_samples` elements from the output
    outputs_to_plot = output[-num_samples:]
    
    # Dynamically calculate the number of rows and columns
    num_cols = min(4, num_samples)  # Limit columns to a maximum of 4
    num_rows = math.ceil(num_samples / num_cols)  # Calculate rows based on samples and columns
    
    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    
    # If there's only one row or column, axes might not be a 2D array
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten in case of multi-dimensional axes for easier iteration
    
    for i, ax in enumerate(axes):
        if i < num_samples:
            current_output = outputs_to_plot[i]
            img_path = current_output.get("img_path", "")
            bounding_boxes = extract_bounding_boxes(current_output.get("output", []))
            boxes = current_output.get("boxes", [])
            label = current_output.get("label", current_output.get("regions", [""])[0])
            
            display_image_with_bboxes(
                ax, 
                img_path,  
                bounding_boxes, 
                boxes, 
                label=label
            )
        else:
            ax.axis('off')  # Hide unused subplots
    
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "images_with_bboxes.png")
    plt.savefig(plot_path)
    plt.close(fig) 
    print(f"Plot saved at {plot_path}")


