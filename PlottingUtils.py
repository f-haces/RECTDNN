import numpy as np
import matplotlib.pyplot as plt
import torch

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

# Function to update the plots
def update_plots(losses, accuracies, fig, axs, 
                 update_interval_x=10,
                 labels=["Training", "Validation"],
                 titles=["Loss", "Intersection over Union"],
                 colors=["b","g","r"]*2,
                 linestyles=["solid", "dotted"]):
    
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
    for i, line in enumerate(axs[1].lines):
        line.set_color(colors[i])
        
    # PRETTIFY
    for i, ax in enumerate(axs):
        ax.legend()
        ax.set_xlim((0, np.ceil(np.array(loss).size / update_interval_x) * update_interval_x ))
        ax.set_title(titles[i])
        ax.set_ylabel(titles[i])
        ax.set_xlabel("Epochs")
        
    # UPDATE FIGURE
    fig.canvas.flush_events() 
    fig.canvas.draw()