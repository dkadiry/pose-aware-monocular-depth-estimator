import os
import cv2
import matplotlib.pyplot as plt
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(f"Sample Image {idx + 1}: {image_file}")
        plt.axis('off')
        
        # Initial guess for cropping (e.g., 100 pixels from bottom)
        plt.axhline(y=image.shape[0] - 120, color='red', linestyle='--')        
        plt.show()
        
        if (idx + 1) % sample_interval == 0:
            user_input = input(f"Displayed {idx + 1} images. Press Enter to continue or type 'exit' to stop: ")
            if user_input.lower() == 'exit':
                break
        

if __name__ == "__main__":
    images_directory = "data/cedarbay/images"  # Update this path as needed
    analyze_cropping(images_directory, num_samples=100, sample_interval=10)
        
