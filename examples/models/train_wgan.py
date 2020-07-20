import tensorflow as tf

from nalp.datasets.image import ImageDataset
from nalp.models import WGAN

# Loading the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Creating an Image Dataset
dataset = ImageDataset(x, batch_size=256, shape=(
    x.shape[0], 28, 28, 1), normalize=True)

# Creating the WGAN
wgan = WGAN(input_shape=(28, 28, 1), noise_dim=100, n_samplings=3,
            alpha=0.3, dropout_rate=0.3, model_type='wc', clip=0.01)

# Compiling the WGAN
wgan.compile(d_optimizer=tf.optimizers.RMSprop(learning_rate=0.00005),
             g_optimizer=tf.optimizers.RMSprop(learning_rate=0.00005))

# Fitting the WGAN
wgan.fit(dataset.batches, epochs=100, critic_steps=5)

# Saving WGAN weights
wgan.save_weights('trained/wgan', save_format='tf')
