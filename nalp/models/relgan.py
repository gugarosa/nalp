# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Relational Generative Adversarial Network."""

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from nalp.core.model import Adversarial
from nalp.encoders.integer import IntegerEncoder
from nalp.models.discriminators.text import TextDiscriminator
from nalp.models.generators.gumbel_rmc import GumbelRMCGenerator


class RelGAN(Adversarial):
    """Train a relational generative adversarial network for text generation."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        max_length: int = 1,
        embedding_size: int = 32,
        n_slots: int = 3,
        n_heads: int = 5,
        head_size: int = 10,
        n_blocks: int = 1,
        n_layers: int = 3,
        n_filters: tuple[int, ...] = (64,),
        filters_size: tuple[int, ...] = (1,),
        dropout_rate: float = 0.25,
        tau: float = 5.0,
    ) -> None:
        """Initialize the text discriminator and Gumbel relational-memory generator.

        Generator calls return vocabulary logits, relaxed probabilities, and token IDs. Relativistic losses
        compare discriminator outputs for real one-hot sequences and generated vocabulary distributions.

        Reference: W. Nie, N. Narodytska, A. Patel.
        Relgan: Relational generative adversarial networks for text generation.
        International Conference on Learning Representations (2018).

        Args:
            encoder: An index to vocabulary encoder for the generator.
            vocab_size: The size of the vocabulary for both discriminator and generator.
            max_length: Maximum length of the sequences for the discriminator.
            embedding_size: The size of the embedding layer for both discriminator and generator.
            n_slots: Number of memory slots for the generator.
            n_heads: Number of attention heads for the generator.
            head_size: Size of each attention head for the generator.
            n_blocks: Number of feed-forward networks for the generator.
            n_layers: Number of layers per feed-forward network for the generator.
            n_filters: Number of filters to be applied in the discriminator.
            filters_size: Size of filters to be applied in the discriminator.
            dropout_rate: Dropout activation rate.
            tau: Gumbel-Softmax temperature parameter.

        Raises:
            ValueError: The Gumbel temperature is non-finite or not positive.

        """

        D = TextDiscriminator(max_length, embedding_size, n_filters, filters_size, dropout_rate)
        G = GumbelRMCGenerator(
            encoder,
            vocab_size,
            embedding_size,
            n_slots,
            n_heads,
            head_size,
            n_blocks,
            n_layers,
            tau,
        )

        super().__init__(D, G, name="RelGAN")

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

        Restore the generator's initial memory and seed each sequence with the first input token.

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
        loss = self.loss(tf.ones_like(y_real), y_real - y_fake)

        return tf.reduce_mean(loss)

    def _generator_loss(self, y_real: tf.Tensor, y_fake: tf.Tensor) -> tf.Tensor:
        loss = self.loss(tf.ones_like(y_fake), y_fake - y_real)

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
        """Update the generator before the discriminator and accumulate both losses.

        Args:
            x: Integer token IDs shaped (batch_size, length), providing each generated sequence's first token.
            y: Integer real target sequences shaped (batch_size, length).

        """

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            _, x_fake_probs = self.generate_batch(x)

            y_fake = self.D(x_fake_probs)

            y = tf.one_hot(y, self.vocab_size)
            y_real = self.D(y)

            G_loss = self._generator_loss(y_real, y_fake)
            D_loss = self._discriminator_loss(y_real, y_fake)

        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))

        self.G_loss.update_state(G_loss)
        self.D_loss.update_state(D_loss)

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
