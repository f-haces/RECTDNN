import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def init_plotting(n_plots=2, titles=["Loss", "Learning Rate"]):

    # Create figure and axes
    fig, axs = plt.subplots(n_plots,1, figsize=(10, 5 * n_plots), dpi=64)
    return fig, axs
    
def calculate_iou(outputs, targets):
    """
    Calculate Intersection over Union (IoU) for semantic segmentation for each class.

    Args:
        outputs (torch.Tensor): Predicted segmentation mask (batch_size, num_classes, height, width).
        targets (torch.Tensor): Ground truth segmentation mask with class indices (batch_size, height, width).

    Returns:
        IoU (torch.Tensor): Intersection over Union scores for each class (num_classes,).
    """
    # Ensure the tensors are on the same device (e.g., CPU or GPU)
    outputs = outputs.to(targets.device)

    # Calculate the number of classes from the outputs tensor
    num_classes = outputs.size(1)

    # Compute the one-hot encoded targets using the class indices
    targets_onehot = torch.zeros_like(outputs)
    targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

    # Flatten the tensors to 2D arrays (batch_size, num_classes, height * width)
    outputs = outputs.view(outputs.size(0), num_classes, -1)
    targets_onehot = targets_onehot.view(targets_onehot.size(0), num_classes, -1)

    # Compute the intersection and union for each class
    intersection = torch.sum(outputs * targets_onehot, dim=2)
    union = torch.sum((outputs + targets_onehot) - (outputs * targets_onehot), dim=2)

    # Calculate IoU for each class
    iou = intersection / (union + 1e-10)  # Add a small epsilon to avoid division by zero

    return iou

def class_to_one_hot(image, num_classes):
    """
    Convert class labels to one-hot encoded vectors.

    Args:
        class_labels (list or numpy.ndarray): List of class labels.
        num_classes (int): Total number of classes.

    Returns:
        numpy.ndarray: One-hot encoded array where each row corresponds to a class label.
    """
    
    one_hot_matrix  = []
    for i in range(num_classes):
        one_hot_matrix.append(np.where(image == i, 1, 0))
        
    one_hot_matrix  = np.dstack(one_hot_matrix)
    

    return one_hot_matrix

# Function to update the plots
def update_plots(losses, accuracies, fig, axs, 
                 update_interval_x=10,
                 labels=["Training", "Validation"],
                 titles=["Loss", "Intersection over Union"],
                 colors=["b","g","r"]*2,
                 linestyles=["solid", "dotted"], 
                 annotations=None):
    
    # IF ONLY ONE LINE PER PLOT, ADD TO UNDERLYING LIST
    if not isinstance(losses[0], list):
        losses = [losses]
    if not isinstance(accuracies[0], list):
        accuracies = [accuracies]
        
    # CLEAR PREVIOUS PLOTS
    for ax in axs:
        ax.cla()
        if ax.get_legend():
            ax.get_legend().remove()
    
    # PLOT
    for i, loss in enumerate(losses):
        axs[0].plot(loss, label=labels[i])
    for i, accuracy in enumerate(accuracies):
        axs[1].plot(accuracy, label=labels[i], linestyle=linestyles[i])
    if colors is not None:
        for i, line in enumerate(axs[1].lines):
            line.set_color(colors[i])
        
    # PRETTIFY
    for i, ax in enumerate(axs):
        ax.legend()
        ax.set_xlim((0, np.ceil(np.array(loss).size / update_interval_x) * update_interval_x ))
        ax.set_title(titles[i])
        ax.set_ylabel(titles[i])
        ax.set_xlabel("Epochs")
        
    if annotations is not None:
        axs[0].set_title(annotations)
    # UPDATE FIGURE
    fig.canvas.flush_events() 
    fig.canvas.draw()
    
def one_hot_to_rgb(one_hot_encoded_image, class_colors=None):
    """
    Converts a one-hot encoded image to a 3-channel RGB image with unique colors for each class.
    
    Args:
        one_hot_encoded_image (numpy.ndarray): The one-hot encoded image.
        class_colors (list of tuples, optional): A list of RGB tuples specifying colors for each class.
            If not provided, random colors will be generated for each class.

    Returns:
        numpy.ndarray: The 3-channel RGB image.
    """
    if class_colors is None:
        # Generate random colors for each class
        num_classes = one_hot_encoded_image.shape[-1]
        class_colors = [tuple(np.random.randint(0, 256, 3)) for _ in range(num_classes)]
    
    # Get the class indices from the one-hot encoded image
    class_indices = np.argmax(one_hot_encoded_image, axis=-1)
    
    # Create the RGB image using the class colors
    rgb_image = np.zeros_like(one_hot_encoded_image, dtype=np.uint8)
    for class_idx, color in enumerate(class_colors):
        mask = class_indices == class_idx
        rgb_image[mask] = color
    
    return rgb_image
    
def probability_to_rgb(probability_image, class_colors=None):
    """
    Converts a PIL image with class probability values to a 3-channel RGB image with unique colors for each class.
    
    Args:
        probability_image (PIL.Image.Image): The PIL image with probability values for each class.
        class_colors (list of tuples, optional): A list of RGB tuples specifying colors for each class.
            If not provided, random colors will be generated for each class.

    Returns:
        PIL.Image.Image: The 3-channel RGB image as a PIL image.
    """
    
    class_colors_master = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255), # Cyan
        (128, 0, 0),   # Maroon
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Navy
        (128, 128, 0), # Olive
        (128, 0, 128), # Purple
        (0, 128, 128), # Teal
    ] * 5
    
    if class_colors is None:
        # Generate random colors for each class
        num_classes = probability_image.shape[-1]
        class_colors = [class_colors_master[i] for i in range(num_classes)]
            
    # Convert PIL image to NumPy array
    probability_array = np.array(probability_image)
    
    # Get the class with the highest probability for each pixel
    class_indices = np.argmax(probability_array, axis=-1)
    
    # Create the RGB image using the class colors
    rgb_array = np.zeros(np.hstack((probability_array.shape[:-1], 3)), dtype=np.uint8)
    for class_idx, color in enumerate(class_colors):
        mask = class_indices == class_idx
        rgb_array[mask] = color
    
    # Convert NumPy array back to PIL image
    rgb_image = Image.fromarray(rgb_array)
    
    return rgb_image