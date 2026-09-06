# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.datasets import ImageDataset
from nalp.models import GAN

(x, y), (_, _) = tf.keras.datasets.mnist.load_data()
dataset = ImageDataset(x, batch_size=256, shape=(x.shape[0], 784), normalize=True)

gan = GAN(input_shape=(784,), noise_dim=100, n_samplings=3, alpha=0.01)
gan.compile(
    d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
)

gan.fit(dataset.batches, epochs=150)

gan.save_weights("trained/gan", save_format="tf")
