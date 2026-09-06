# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import matplotlib.pyplot as plt
import tensorflow as tf

from nalp.models import GAN

# Match the training data, model class, and parameters when restoring weights
gan = GAN(input_shape=(784,), noise_dim=100, n_samplings=3, alpha=0.01)
gan.load_weights("trained/gan").expect_partial()

z = tf.random.normal([16, 1, 1, 100])
sampled_images = tf.reshape(gan.G(z), (16, 28, 28))

fig = plt.figure(figsize=(4, 4))
for i in range(sampled_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(sampled_images[i, :, :] * 127.5 + 127.5, cmap="gray")
    plt.axis("off")

plt.show()
