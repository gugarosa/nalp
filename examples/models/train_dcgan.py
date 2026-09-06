# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.datasets import ImageDataset
from nalp.models import DCGAN

(x, y), (_, _) = tf.keras.datasets.mnist.load_data()
dataset = ImageDataset(x, batch_size=256, shape=(x.shape[0], 28, 28, 1), normalize=True)

dcgan = DCGAN(input_shape=(28, 28, 1), noise_dim=100, n_samplings=3, alpha=0.3, dropout_rate=0.3)

dcgan.compile(
    d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
)

dcgan.fit(dataset.batches, epochs=100)

dcgan.save_weights("trained/dcgan", save_format="tf")
