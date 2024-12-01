import os
import cv2
import numpy as np
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tools import get_images_from_directory

def compute_average_crop(images_dir: str, num_samples: int = 100) -> int:
    """
    Computes the average number of pixels to crop from the bottom to remove the rover platform.
    
    Parameters:
    - images_dir (str): Path to the images directory.
    - num_samples (int): Number of images to analyze.
    
    Returns:
    - int: Average number of pixels to crop.
    """
    image_files = get_images_from_directory(images_dir)[:num_samples]
    crop_pixels_list = []
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assume the largest contour at the bottom is the rover platform
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            crop_pixels = image.shape[0] - (y + h)
            crop_pixels_list.append(crop_pixels)
        else:
            # Default crop if no contours found
            crop_pixels_list.append(100)
    
    average_crop = int(np.mean(crop_pixels_list))
    print(f"Average crop pixels determined: {average_crop}")
    return average_crop

if __name__ == "__main__":
    images_directory = "data/cedarbay/images"  # Update this path as needed
    compute_average_crop(images_directory, num_samples=100)
    