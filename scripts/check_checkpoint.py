import tensorflow as tf

def list_checkpoint_variables(checkpoint_path):
    variables = tf.train.list_variables(checkpoint_path)
    print(f"Variables in checkpoint '{checkpoint_path}':")
    for var_name, shape in variables:
        print(f"{var_name}: {shape}")

# Example usage
checkpoint_path = 'path/to/your/checkpoint'  # Replace with your actual checkpoint path
list_checkpoint_variables(checkpoint_path)