import tensorflow as tf 
from utils.tools import load_config
import os
from models.unet import DepthEstimationModel, DownscaleBlock, UpscaleBlock, BottleNeckBlock
import numpy as np

def main():

    config_path = 'config/inference.yaml'
    if not os.path.exists(config_path):
        print(f"Inference configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Extract parameters
    model_params = config['model_parameters']
    inference_params = config['inference_parameters']

    # Select model variant for inference
    model_variant = inference_params['model_variant']
    if model_variant not in model_params['input_shapes']:
        print(f"Invalid model_variant: {model_variant}. Available options: {list(model_params['input_shapes'].keys())}")
        return
    
    # Initialize model based on model_variant 'vanilla' 'rel_z' or 'rel_z_pitch_roll'
    input_shape = model_params['input_shapes'][model_variant] 
    input_channels = input_shape[2]
    model = DepthEstimationModel(
        width=input_shape[1],
        height=input_shape[0],
        input_channels=input_channels
    )
    
    # Build the model to initialize weights
    model.build(input_shape=(None, input_shape[0], input_shape[1], input_channels))
    model.summary()

    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    #model.compile(optimizer=optimizer)
    #print("Model compiled successfully.")
    
    # Load the trained model weights
    checkpoint_file = inference_params['models'][model_variant]['checkpoint_file']
    if not os.path.exists(checkpoint_file + '.index'):  # TensorFlow saves .index and .data files
        print(f"Checkpoint file not found: {checkpoint_file}")
        return
    
    try:
        # Load weights with 'by_name=True' to match layers by name
        # 'skip_mismatch=True' allows layers with mismatched shapes to skip loading
        status = model.load_weights(checkpoint_file)
        status.expect_partial()  # Suppress warnings about missing variables like optimizer states
        print(f"Loaded best weights from {checkpoint_file}")
    except Exception as e:
        print(f"Error loading weights from checkpoint: {e}")
        return
    
    try:
        # Create dummy input data with the correct shape
        dummy_input = np.random.rand(1, input_shape[0], input_shape[1], input_channels).astype(np.float32)
        dummy_output = model(dummy_input)
        print(f"Dummy output shape: {dummy_output.shape}")  # Expected: (1, height, width, 1)
    except Exception as e:
        print(f"Error during forward pass: {e}")


    # Save the complete model (architecture + weights)
    save_model_dir = inference_params['models'][model_variant]['save_best_model_path']
    try:
        os.makedirs(save_model_dir, exist_ok=True)
        model.save(save_model_dir)
        print(f"Best model saved to {save_model_dir}")
    except Exception as e:
        print(f"Error saving the best model: {e}")
        return

if __name__ == "__main__":
    main()