import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Dict, Any

def display_depth(depth_map: np.ndarray, cmap: str = 'plasma', title: str = "Depth Map") -> None:
    """
    Displays a single depth map.
    
    Parameters:
    - depth_map (np.ndarray): Depth map to display.
    - cmap (str): Colormap for visualization.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap=cmap)
    plt.colorbar(label="Depth")
    plt.title(title)
    plt.axis('off')
    plt.show()

def overlay_depth_on_image(image: np.ndarray, depth_map: np.ndarray, alpha: float = 0.8, cmap: str = 'plasma') -> np.ndarray:
    """
    Overlays a depth map onto an RGB image.
    
    Parameters:
    - image (np.ndarray): Original RGB image in [0, 255].
    - depth_map (np.ndarray): Normalized depth map in [0, 1].
    - alpha (float): Transparency factor.
    - cmap (str): Colormap for depth map.
    
    Returns:
    - np.ndarray: Image with depth overlay.
    """

    depth_colored = plt.get_cmap(cmap)(depth_map)[:, :, :3]  # [0,1]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Ensure image is uint8
    image_uint8 = image.astype(np.uint8)
    
    # Blend images
    overlayed_image = cv2.addWeighted(image_uint8, 1 - alpha, depth_colored, alpha, 0)
    return overlayed_image

def plot_depth_histogram(depth_map: np.ndarray, bins: int = 50, title: str = "Depth Histogram") -> None:
    """
    Plots a histogram of depth values.
    
    Parameters:
    - depth_map (np.ndarray): Depth map to analyze.
    - bins (int): Number of histogram bins.
    - title (str): Title of the histogram.
    """

    plt.figure(figsize=(8, 6))
    plt.hist(depth_map.flatten(), bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def calculate_depth_error(gt_depth_map: np.ndarray, predicted_depth_map: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates the error map and Mean Squared Error (MSE).
    
    Parameters:
    - gt_depth_map (np.ndarray): Ground truth depth map.
    - predicted_depth_map (np.ndarray): Predicted depth map.
    
    Returns:
    - Tuple[np.ndarray, float]: Error map and MSE.
    """
    error_map = (predicted_depth_map - gt_depth_map).astype(np.float64)
    non_zero = np.count_nonzero(gt_depth_map)
    mse = np.sum(error_map**2) / non_zero if non_zero != 0 else float('inf')
    return error_map, mse

def create_error_map(error_map: np.ndarray, cmap: str = 'plasma', title: str = "Depth Error Map") -> None:
    """
    Displays the depth error map.
    
    Parameters:
    - error_map (np.ndarray): Error map to display.
    - cmap (str): Colormap for visualization.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(error_map, cmap=cmap)
    plt.colorbar(label='Error')
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_error_map(error_map: np.ndarray, save_path: str, cmap: str = 'plasma') -> None:
    """
    Saves the error map as an image.
    
    Parameters:
    - error_map (np.ndarray): Error map to save.
    - save_path (str): File path to save the image.
    - cmap (str): Colormap for visualization.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(error_map, cmap=cmap)
    plt.colorbar(label='Error')
    plt.title("Depth Error Map")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_sample(image: np.ndarray, depth_map: np.ndarray, mask: np.ndarray, cmap: str = 'plasma', alpha: float = 0.6,
                    title_image: str = "RGB Image",
                    title_depth: str = "Depth Map",
                    title_mask: str = "Depth Mask",
                    title_overlay: str = "Depth Overlay",
                    mode: str = "all") -> None:
    """
    Visualizes the image, depth map, and their overlay depending on the mode selected.
    """

    if mode == "all":
        if image is None or depth_map is None:
            raise ValueError("Both image and depth_map must be provided for 'all' mode.")
        overlayed_image = overlay_depth_on_image(image, depth_map, alpha=alpha, cmap=cmap)
        
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(title_image)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(depth_map, cmap=cmap)
        plt.title(title_depth)
        plt.axis('off')
        plt.colorbar(label='Depth')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlayed_image)
        plt.title(title_overlay)
        plt.axis('off')
        
        plt.show()
    
    elif mode == "depth_only":
        if depth_map is None:
            raise ValueError("depth_map must be provided for 'depth_only' mode.")
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_map, cmap=cmap)
        plt.colorbar(label="Depth")
        plt.title(title_depth)
        plt.axis('off')
        plt.show()
    
    elif mode == "image_only":
        if image is None:
            raise ValueError("image must be provided for 'image_only' mode.")
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(title_image)
        plt.axis('off')
        plt.show()

    elif mode == "mask_only":
        if mask is None:
            raise ValueError("mask must be provided for 'mask_only' mode.")
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray')
        plt.colorbar(label="Mask")
        plt.title(title_mask)
        plt.axis('off')
        plt.show()

    elif mode == "all_with_mask":
        if image is None or depth_map is None or mask is None:
            raise ValueError("Image, depth_map, and mask must be provided for 'all_with_mask' mode.")
        overlayed_image = overlay_depth_on_image(image, depth_map, alpha=alpha, cmap=cmap)
        
        plt.figure(figsize=(24, 6))
        
        plt.subplot(1, 4, 1)      
        plt.imshow(image)
        plt.title(title_image)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(depth_map, cmap=cmap)
        plt.title(title_depth)
        plt.axis('off')
        plt.colorbar(label='Depth')
        
        plt.subplot(1, 4, 3)
        plt.imshow(mask, cmap='gray')
        plt.title(title_mask)
        plt.axis('off')
        plt.colorbar(label='Mask')
        
        plt.subplot(1, 4, 4)
        plt.imshow(overlayed_image)
        plt.title(title_overlay)
        plt.axis('off')
        
        plt.show()


    else:
        raise ValueError("Invalid mode selected. Choose from 'all', 'depth_only', or 'image_only'.")
