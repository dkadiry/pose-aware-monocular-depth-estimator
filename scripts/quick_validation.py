# quick_validation.py

import tensorflow as tf
import numpy as np
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.unet import DepthEstimationModel, DownscaleBlock, UpscaleBlock, BottleNeckBlock
from utils.tools import load_config, denormalize_depth_map_global

def main():
    # Load configuration
    config_path = 'config/inference.yaml'
    if not os.path.exists(config_path):
        print(f"Inference configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    dataset_params = config['dataset_parameters']
    inference_params = config['inference_parameters']
    
    model_variant = inference_params['model_variant']
    saved_model_path = inference_params['models'][model_variant]['saved_model_path']
    
    # Load the SavedModel
    try:
        model = tf.keras.models.load_model(
            saved_model_path,
            custom_objects={
                'DepthEstimationModel': DepthEstimationModel,
                'DownscaleBlock': DownscaleBlock,
                'UpscaleBlock': UpscaleBlock,
                'BottleNeckBlock': BottleNeckBlock
            }
        )
        print(f"Loaded model from {saved_model_path}")
        model.summary()
    except Exception as e:
        print(f"Error loading SavedModel: {e}")
        return
    
    # Create a dummy input matching the model's expected input shape
    input_shape = (None, 768, 1024, 3)  # Should be (None, height, width, channels)
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]
    
    dummy_input = np.random.rand(1, height, width, channels).astype(np.float32)
    
    # Make a prediction
    prediction = model.predict(dummy_input)
    print(f"Dummy prediction shape: {prediction.shape}")
    print(f"Dummy prediction max: {prediction.max()}, min: {prediction.min()}")
    
    # Denormalize if necessary
    percentiles = np.load(dataset_params['percentiles_path'])
    lower_global = percentiles['P1']
    upper_global = percentiles['P99']
    
    denorm_pred = denormalize_depth_map_global(prediction, lower_global, upper_global)
    print(f"Denormalized prediction max: {denorm_pred.max()}, min: {denorm_pred.min()}")

if __name__ == "__main__":
    main()