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

def overlay_depth_on_image(image: np.ndarray, depth_map: np.ndarray, alpha: float = 0.6, cmap: str = 'plasma') -> np.ndarray:
    """
    Overlays a depth map onto an RGB image.
    
    Parameters:
    - image (np.ndarray): Original RGB image.
    - depth_map (np.ndarray): Depth map to overlay.
    - alpha (float): Transparency factor.
    - cmap (str): Colormap for depth map.
    
    Returns:
    - np.ndarray: Image with depth overlay.
    """
    depth_normalized = depth_map / np.max(depth_map) if np.max(depth_map) != 0 else depth_map
    depth_colored = plt.get_cmap(cmap)(depth_normalized)[:, :, :3]
    overlayed_image = (1 - alpha) * image + alpha * (depth_colored * 255).astype(np.uint8)
    return overlayed_image.astype(np.uint8)

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

def visualize_sample(image: np.ndarray, depth_map: np.ndarray, cmap: str = 'plasma', alpha: float = 0.6,
                    title_image: str = "RGB Image",
                    title_depth: str = "Depth Map",
                    title_overlay: str = "Depth Overlay") -> None:
    """
    Visualizes the image, depth map, and their overlay.
    
    Parameters:
    - image (np.ndarray): RGB image.
    - depth_map (np.ndarray): Depth map.
    - cmap (str): Colormap for depth map.
    - alpha (float): Transparency for overlay.
    - title_image (str): Title for image subplot.
    - title_depth (str): Title for depth map subplot.
    - title_overlay (str): Title for overlay subplot.
    """
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