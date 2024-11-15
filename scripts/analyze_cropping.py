import os
import cv2
import matplotlib.pyplot as plt
from utils.tools import get_images_from_directory

def analyze_cropping(images_dir: str, num_samples: int = 100, sample_interval: int = 10):
    """
    Loads and displays sample images to determine cropping pixels.
    
    Parameters:
    - images_dir (str): Path to the images directory.
    - num_samples (int): Number of images to analyze.
    - sample_interval (int): Interval at which to pause for user input.
    """
     
    image_files = get_images_from_directory(images_dir)[:num_samples]

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
