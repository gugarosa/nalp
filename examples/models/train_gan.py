import tensorflow as tf

from nalp.datasets import ImageDataset
from nalp.models import GAN

# Loading the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Creating an Image Dataset
dataset = ImageDataset(x, batch_size=256, shape=(x.shape[0], 784), normalize=True)

# Creating the GAN
gan = GAN(input_shape=(784,), noise_dim=100, n_samplings=3, alpha=0.01)

# Compiling the GAN
gan.compile(
    d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
)

# Fitting the GAN
gan.fit(dataset.batches, epochs=150)

# Saving GAN weights
gan.save_weights("trained/gan", save_format="tf")
