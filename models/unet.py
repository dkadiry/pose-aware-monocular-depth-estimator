import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore


class DepthEstimationModel(tf.keras.Model):
    """U-Net Architecture for Depth Estimation."""

    def __init__(self, width, height, input_channels=3, name="DepthEstimationModel"):
        super().__init__(name=name)
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.width = width
        self.height = height
        self.input_channels = input_channels

        # Define filter sizes for different blocks
        f = [16, 32, 64, 128, 256]

        # Encoder: Downscale Blocks
        self.downscale_blocks = [
            DownscaleBlock(f[0], name=f"downscale_block_{0}"),
            DownscaleBlock(f[1], name=f"downscale_block_{1}"),
            DownscaleBlock(f[2], name=f"downscale_block_{2}"),
            DownscaleBlock(f[3], name=f"downscale_block_{3}")

        ]

        # Bottleneck Block
        self.bottle_neck_block = BottleNeckBlock(f[4], name="bottleneck_block")

        # Decoder: Upscale Blocks
        self.upscale_blocks = [
            UpscaleBlock(f[3], name=f"upscale_block_{0}"),
            UpscaleBlock(f[2], name=f"upscale_block_{1}"),
            UpscaleBlock(f[1], name=f"upscale_block_{2}"),
            UpscaleBlock(f[0], name=f"upscale_block_{3}"),
        ]

        # Final Convolutional Layer for Depth Output
        self.conv_layer = layers.Conv2D(
            1, (1, 1), padding="same", activation="relu", name="final_conv"
        )

    def calculate_loss(self, target, pred):
        """Calculate the combined loss."""
        # Compute image gradients
        dy_true, dx_true = tf.image.image_gradients(target)
        dy_pred, dx_pred = tf.image.image_gradients(pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness loss
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y
        depth_smoothness_loss = tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(
            tf.abs(smoothness_y)
        )

        # Structural Similarity (SSIM) loss
        ssim_loss = tf.reduce_mean(
            1 - tf.image.ssim(
                target, pred, max_val=1.0, filter_size=11, k1=0.01, k2=0.03
            )
        )

        # L1 loss (Point-wise depth)
        l1_loss = tf.reduce_mean(tf.abs(target - pred))

        # Combined loss
        loss = (
            (self.ssim_loss_weight * ssim_loss)
            + (self.l1_loss_weight * l1_loss)
            + (self.edge_loss_weight * depth_smoothness_loss)
        )

        return loss

    @property
    def metrics(self):
        """List of metrics to track."""
        return [self.loss_metric]

    def train_step(self, batch_data):
        """Custom training step."""
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = self.calculate_loss(target, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def test_step(self, batch_data):
        """Custom testing step."""
        input, target = batch_data

        pred = self(input, training=False)
        loss = self.calculate_loss(target, pred)

        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def call(self, x):
        """Forward pass."""
        # Encoder pathway
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        # Bottleneck
        bn = self.bottle_neck_block(p4)

        # Decoder pathway with skip connections
        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)

        # Final output layer
        return self.conv_layer(u4)
   
    def get_config(self):
        """
        Returns the config of the model. This is used for serialization.
        """
        config = super(DepthEstimationModel, self).get_config()
        config.update({
            'width': self.width,
            'height': self.height,
            'input_channels': self.input_channels,
            
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a model from its config. This is used for deserialization.
        """
        return cls(**config)


class DownscaleBlock(layers.Layer):
    """Downscaling block with two convolutions and residual connection."""

    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters  # Store as instance variable
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        """Forward pass."""
        # First convolution
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        # Second convolution
        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        # Residual connection
        x += d

        # Pooling
        p = self.pool(x)
        return x, p
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpscaleBlock(layers.Layer):
    """Upscaling block with upsampling, concatenation, and two convolutions."""

    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters  # Store as instance variable
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        """Forward pass."""
        # Upsampling
        x = self.us(x)

        # Concatenate with skip connection from encoder
        concat = self.conc([x, skip])

        # First convolution
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        # Second convolution
        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BottleNeckBlock(layers.Layer):
    """Bottleneck block with two convolutions."""

    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters  # Store as instance variable
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        """Forward pass."""
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_depth_estimation_model(width, height):
    """Helper function to build the DepthEstimationModel."""
    return DepthEstimationModel(width=width, height=height)


if __name__ == "__main__":
    # Example usage: Instantiate the model
    width = 1024
    height = 768  
    input_channels = 3 # 3, 4, or 6 depending on model

    model = DepthEstimationModel(width=width, height=height, input_channels=input_channels)
    model.build(input_shape=(None, height, width, input_channels))
    model.summary()
