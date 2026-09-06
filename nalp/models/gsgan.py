# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Gumbel-Softmax Generative Adversarial Network."""

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from nalp.core.model import Adversarial
from nalp.encoders.integer import IntegerEncoder
from nalp.models.discriminators.lstm import LSTMDiscriminator
from nalp.models.generators.gumbel_lstm import GumbelLSTMGenerator


class GSGAN(Adversarial):
    """Train a Gumbel-Softmax generative adversarial network for discrete sequences."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        tau: float = 5,
    ) -> None:
        """Initialize the stateful LSTM discriminator and Gumbel generator.

        Generator calls return vocabulary logits, relaxed probabilities, and token IDs. The discriminator
        consumes vocabulary distributions and retains its recurrent state across calls.

        Reference: M. Kusner, J. Hernández-Lobato.
        Gans for sequences of discrete elements with the gumbel-softmax distribution.
        Preprint arXiv:1611.04051 (2016).

        Args:
            encoder: An index to vocabulary encoder for the generator.
            vocab_size: The size of the vocabulary for both discriminator and generator.
            embedding_size: The size of the embedding layer for both discriminator and generator.
            hidden_size: The amount of hidden neurons for the generator.
            tau: Gumbel-Softmax temperature parameter.

        Raises:
            ValueError: The Gumbel temperature is non-finite or not positive.

        """

        D = LSTMDiscriminator(embedding_size, hidden_size)
        G = GumbelLSTMGenerator(encoder, vocab_size, embedding_size, hidden_size, tau)

        super().__init__(D, G, name="GSGAN")

        self.vocab_size = vocab_size
        self.init_tau = tau

    def compile(
        self,
        pre_optimizer: tf.keras.optimizers.Optimizer,
        d_optimizer: tf.keras.optimizers.Optimizer,
        g_optimizer: tf.keras.optimizers.Optimizer,
    ) -> None:
        """Configure optimizers and reset pre-training and adversarial loss tracking.

        Args:
            pre_optimizer: An optimizer instance for pre-training the generator.
            d_optimizer: An optimizer instance for the discriminator.
            g_optimizer: An optimizer instance for the generator.

        """

        self.P_optimizer = pre_optimizer
        self.D_optimizer = d_optimizer
        self.G_optimizer = g_optimizer

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits
        self.D_loss = tf.metrics.Mean(name="D_loss")
        self.G_loss = tf.metrics.Mean(name="G_loss")

        self.history["pre_G_loss"] = []
        self.history["D_loss"] = []
        self.history["G_loss"] = []

    def generate_batch(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate autoregressive tokens and their Gumbel-Softmax probabilities.

        Reset generator state and seed each sequence with the first input token.

        Args:
            x: Integer token IDs shaped (batch_size, length) with fixed batch and sequence dimensions.

        Returns:
            Generated IDs shaped (batch_size, length) and probabilities shaped (batch_size, length, vocab_size).

        """

        batch_size, max_length = x.shape[0], x.shape[1]

        start_batch = tf.expand_dims(x[:, 0], -1)

        sampled_preds = tf.zeros([batch_size, 0, self.vocab_size])
        sampled_batch = start_batch

        self.G.reset_state()

        for _ in range(max_length):
            _, preds, start_batch = self.G(start_batch)

            sampled_preds = tf.concat([sampled_preds, preds], 1)
            sampled_batch = tf.concat([sampled_batch, start_batch], 1)

        sampled_batch = sampled_batch[:, 1:]

        return sampled_batch, sampled_preds

    def _discriminator_loss(self, y_real: tf.Tensor, y_fake: tf.Tensor) -> tf.Tensor:
        real_loss = self.loss(tf.ones_like(y_real), y_real)
        fake_loss = self.loss(tf.zeros_like(y_fake), y_fake)

        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    def _generator_loss(self, y_fake: tf.Tensor) -> tf.Tensor:
        loss = self.loss(tf.ones_like(y_fake), y_fake)

        return tf.reduce_mean(loss)

    @tf.function
    def G_pre_step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Pre-train the generator on next-token logits and accumulate its loss.

        Args:
            x: Integer input token IDs shaped (batch_size, length).
            y: Integer next-token targets shaped (batch_size, length).

        """

        with tf.GradientTape() as tape:
            logits, _, _ = self.G(x)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits))

        gradients = tape.gradient(loss, self.G.trainable_variables)

        self.P_optimizer.apply_gradients(zip(gradients, self.G.trainable_variables))

        self.G_loss.update_state(loss)

    @tf.function
    def step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Update the discriminator before the generator and accumulate both losses.

        Args:
            x: Integer token IDs shaped (batch_size, length), providing each generated sequence's first token.
            y: Integer real target sequences shaped (batch_size, length).

        """

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            _, x_fake_probs = self.generate_batch(x)

            y_fake = self.D(x_fake_probs)

            y = tf.one_hot(y, self.vocab_size)
            y_real = self.D(y)

            D_loss = self._discriminator_loss(y_real, y_fake)
            G_loss = self._generator_loss(y_fake)

        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)

        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))
        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))

        self.D_loss.update_state(D_loss)
        self.G_loss.update_state(G_loss)

    def pre_fit(self, batches: tf.data.Dataset, epochs: int = 100) -> None:
        """Pre-train the generator and append its per-epoch loss to history.

        Args:
            batches: Finite dataset of (input_ids, target_ids) batches shaped (batch_size, length).
            epochs: The maximum number of pre-training epochs.

        """

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for _ in range(epochs):
            self.G_loss.reset_state()

            b = Progbar(n_batches, stateful_metrics=["loss(G)"])

            for x_batch, y_batch in batches:
                self.G_pre_step(x_batch, y_batch)

                b.add(1, values=[("loss(G)", self.G_loss.result())])

            self.history["pre_G_loss"].append(self.G_loss.result().numpy())

    def fit(self, batches: tf.data.Dataset, epochs: int = 100) -> None:
        """Train adversarially and append per-epoch generator and discriminator losses.

        Reset loss metrics each epoch and exponentially anneal the generator's Gumbel temperature afterward.

        Args:
            batches: Finite dataset of (input_ids, target_ids) batches with fixed batch and sequence dimensions.
            epochs: The maximum number of training epochs.

        """

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for e in range(epochs):
            self.G_loss.reset_state()
            self.D_loss.reset_state()

            b = Progbar(n_batches, stateful_metrics=["loss(G)", "loss(D)"])

            for x_batch, y_batch in batches:
                self.step(x_batch, y_batch)

                b.add(
                    1,
                    values=[
                        ("loss(G)", self.G_loss.result()),
                        ("loss(D)", self.D_loss.result()),
                    ],
                )

            self.G.tau = self.init_tau ** ((epochs - e) / epochs)

            self.history["G_loss"].append(self.G_loss.result().numpy())
            self.history["D_loss"].append(self.D_loss.result().numpy())
