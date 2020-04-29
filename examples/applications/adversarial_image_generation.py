import matplotlib.pyplot as plt
import tensorflow as tf

from nalp.models import GAN

# When generating artificial images, make sure
# to use the same data, classes and parameters
# as the pre-trained network

# Creating the GAN
gan = GAN(input_shape=(784,), noise_dim=100, n_samplings=3, alpha=0.01)

# Loading pre-trained GAN weights
gan.load_weights('trained/gan').expect_partial()

# Creating a noise tensor for further sampling
z = tf.random.normal([16, 1, 1, 100])

# Sampling an artificial image
sampled_images = tf.reshape(gan.G(z), (16, 28, 28))

# Creating a pyplot figure
fig = plt.figure(figsize=(4,4))

# For every possible generated image
for i in range(sampled_images.shape[0]):
    # Defines the subplot
    plt.subplot(4, 4, i+1)

    # Plots the image to the figure
    plt.imshow(sampled_images[i, :, :] * 127.5 + 127.5, cmap='gray')

    # Disabling the axis
    plt.axis('off')

# Showing the plot
plt.show()
