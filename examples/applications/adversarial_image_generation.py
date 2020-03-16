import matplotlib.pyplot as plt
import tensorflow as tf

from nalp.models.adversarial.gan import GAN

# When generating artificial images, make sure
# to use the same data, classes and parameters
# as the pre-trained network

# Defining a constant to hold the amount of samples to generate
N_SAMPLES = 16

# Defining a constant to hold the input noise dimension
N_NOISE = 100

# Defining a constant to hold the number of features
N_FEATURES = 784

# Creating the GAN
gan = GAN(gen_input=N_NOISE, gen_output=N_FEATURES, alpha=0.01)

# Loading pre-trained GAN weights
gan.load_weights('trained/gan').expect_partial()

# Creating a noise tensor for further sampling
z = tf.random.normal([16, N_NOISE])

# Sampling an artificial image
sampled_images = tf.reshape(gan.sample(z), (N_SAMPLES, 28, 28))

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
