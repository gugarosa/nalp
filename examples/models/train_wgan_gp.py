# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.datasets import ImageDataset
from nalp.models import WGAN

(x, y), (_, _) = tf.keras.datasets.mnist.load_data()
dataset = ImageDataset(x, batch_size=256, shape=(x.shape[0], 28, 28, 1), normalize=True)

wgan = WGAN(
    input_shape=(28, 28, 1),
    noise_dim=100,
    n_samplings=3,
    alpha=0.3,
    dropout_rate=0.3,
    model_type="gp",
    penalty=10,
)

wgan.compile(
    d_optimizer=tf.optimizers.RMSprop(learning_rate=0.00005),
    g_optimizer=tf.optimizers.RMSprop(learning_rate=0.00005),
)

wgan.fit(dataset.batches, epochs=100, critic_steps=5)

wgan.save_weights("trained/wgan_gp", save_format="tf")
