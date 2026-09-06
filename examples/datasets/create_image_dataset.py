# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.datasets import ImageDataset

(x, y), (_, _) = tf.keras.datasets.mnist.load_data()
dataset = ImageDataset(x, batch_size=256, shape=(x.shape[0], 784), normalize=True, shuffle=True)

for input_batch in dataset.batches.take(1):
    for x in input_batch:
        print(x.numpy())
