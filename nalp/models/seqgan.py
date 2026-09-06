# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Sequence Generative Adversarial Network."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

import nalp.utils.constants as c
from nalp.core.model import Adversarial
from nalp.encoders.integer import IntegerEncoder
from nalp.models.discriminators.embedded_text import EmbeddedTextDiscriminator
from nalp.models.generators.lstm import LSTMGenerator


class SeqGAN(Adversarial):
    """Train a sequence generative adversarial network with Monte Carlo policy rewards."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        max_length: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        n_filters: tuple[int, ...] = (64,),
        filters_size: tuple[int, ...] = (1,),
        dropout_rate: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the embedded-text discriminator and stateful LSTM generator.

        Monte Carlo rollouts estimate each generated prefix's reward from completed sequences.
        Real samples use discriminator label 0 and generated samples use label 1.

        Reference: L. Yu, et al. Seqgan: Sequence generative adversarial nets with policy gradient.
        31st AAAI Conference on Artificial Intelligence (2017).

        Args:
            encoder: An index to vocabulary encoder for the generator.
            vocab_size: The size of the vocabulary for both discriminator and generator.
            max_length: Maximum length of the sequences for the discriminator.
            embedding_size: The size of the embedding layer for both discriminator and generator.
            hidden_size: The amount of hidden neurons for the generator.
            n_filters: Number of filters to be applied in the discriminator.
            filters_size: Size of filters to be applied in the discriminator.
            dropout_rate: Dropout activation rate.
            temperature: Temperature value to sample the token.

        """

        D = EmbeddedTextDiscriminator(
            vocab_size,
            max_length,
            embedding_size,
            n_filters,
            filters_size,
            dropout_rate,
        )
        G = LSTMGenerator(encoder, vocab_size, embedding_size, hidden_size)

        super().__init__(D, G, name="seqgan")

        self.vocab_size = vocab_size
        self.T = temperature

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

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits
        self.D_loss = tf.metrics.Mean(name="D_loss")
        self.G_loss = tf.metrics.Mean(name="G_loss")

        self.history["pre_D_loss"] = []
        self.history["pre_G_loss"] = []
        self.history["D_loss"] = []
        self.history["G_loss"] = []

    def generate_batch(self, batch_size: int = 1, length: int = 1) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate autoregressive input contexts and next-token targets.

        Reset generator state and start each sequence from a uniformly sampled vocabulary ID.
        Sample subsequent IDs from logits divided by the configured temperature.

        Args:
            batch_size: Size of the batch to be generated.
            length: Number of next-token targets to generate per sequence.

        Returns:
            Input context and next-token targets, each an int32 tensor shaped (batch_size, length).

        """

        start_batch = tf.random.uniform([batch_size, 1], 0, self.vocab_size, dtype="int32")
        sampled_batch = start_batch

        self.G.reset_state()

        for _ in range(length):
            preds = self.G(start_batch)

            preds = tf.squeeze(preds, 1)
            preds /= self.T

            start_batch = tf.random.categorical(preds, 1, dtype="int32")
            sampled_batch = tf.concat([sampled_batch, start_batch], 1)

        x_sampled_batch = sampled_batch[:, :length]
        y_sampled_batch = sampled_batch[:, 1:]

        return x_sampled_batch, y_sampled_batch

    def _get_reward(
        self,
        x: tf.Tensor,
        n_rollouts: int,
        start_tokens: tf.Tensor | None = None,
    ) -> tf.Tensor:
        if n_rollouts < 1:
            raise ValueError(f"`n_rollouts` must be positive, but got {n_rollouts}.")

        max_length = x.shape[1]
        rewards = []

        for _ in range(n_rollouts):
            rollout_rewards = []
            for step in range(1, max_length + 1):
                self.G.reset_state()
                samples = x[:, :step]

                if step < max_length:
                    context = samples
                    if start_tokens is not None:
                        context = tf.concat([start_tokens, context], axis=1)
                    output = self.G(context)[:, -1, :]

                    for _ in range(step, max_length):
                        token = tf.random.categorical(output / self.T, 1, dtype="int32")
                        samples = tf.concat([samples, token], axis=1)
                        output = tf.squeeze(self.G(token), 1)

                output = tf.squeeze(tf.nn.softmax(self.D(samples)), 1)
                rollout_rewards.append(output[:, 0])

            rewards.append(tf.stack(rollout_rewards, axis=1))

        return tf.reduce_mean(tf.stack(rewards, axis=0), axis=0)

    @tf.function
    def G_pre_step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Pre-train the generator on next-token logits and accumulate its loss.

        Args:
            x: Integer input token IDs shaped (batch_size, length).
            y: Integer next-token targets shaped (batch_size, length).

        """

        with tf.GradientTape() as tape:
            preds = self.G(x)

            loss = tf.reduce_mean(self.loss(y, preds))

        gradients = tape.gradient(loss, self.G.trainable_variables)

        self.P_optimizer.apply_gradients(zip(gradients, self.G.trainable_variables))

        self.G_loss.update_state(loss)

    @tf.function
    def G_step(self, x: tf.Tensor, y: tf.Tensor, rewards: tf.Tensor) -> None:
        """Reset recurrent state and update the generator with reward-weighted loss.

        Args:
            x: Integer input token IDs shaped (batch_size, length).
            y: Integer next-token targets shaped (batch_size, length).
            rewards: Monte Carlo token rewards shaped (batch_size, length).

        """

        self.G.reset_state()

        with tf.GradientTape() as tape:
            preds = self.G(x)

            loss = tf.reduce_mean(self.loss(y, preds) * rewards)

        gradients = tape.gradient(loss, self.G.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients, self.G.trainable_variables))

        self.G_loss.update_state(loss)

    @tf.function
    def D_step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Update the discriminator once and accumulate its classification loss.

        Args:
            x: Integer token sequences shaped (batch_size, length).
            y: Integer labels shaped (batch_size,), with 0 for real sequences and 1 for generated sequences.

        """

        with tf.GradientTape() as tape:
            preds = tf.squeeze(self.D(x), 1)

            loss = tf.reduce_mean(self.loss(y, preds))

        gradients = tape.gradient(loss, self.D.trainable_variables)

        self.D_optimizer.apply_gradients(zip(gradients, self.D.trainable_variables))

        self.D_loss.update_state(loss)

    def pre_fit(
        self,
        batches: tf.data.Dataset,
        g_epochs: int = 50,
        d_epochs: int = 10,
    ) -> None:
        """Pre-train the generator before the discriminator and append their losses to history.

        Args:
            batches: Finite dataset of (input_ids, target_ids) batches with fixed batch and sequence dimensions.
            g_epochs: The maximum number of pre-training generator epochs.
            d_epochs: The maximum number of pre-training discriminator epochs.

        """

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for _ in range(g_epochs):
            self.G_loss.reset_state()

            b = Progbar(n_batches, stateful_metrics=["loss(G)"])

            for x_batch, y_batch in batches:
                self.G_pre_step(x_batch, y_batch)

                b.add(1, values=[("loss(G)", self.G_loss.result())])

            self.history["pre_G_loss"].append(self.G_loss.result().numpy())

        for _ in range(d_epochs):
            self.D_loss.reset_state()

            b = Progbar(n_batches, stateful_metrics=["loss(D)"])

            for _, y_batch in batches:
                batch_size, max_length = y_batch.shape[0], y_batch.shape[1]

                _, y_fake_batch = self.generate_batch(batch_size, max_length)

                x_concat_batch = tf.concat([y_batch, y_fake_batch], 0)
                y_concat_batch = tf.concat(
                    [
                        tf.zeros(batch_size, dtype="int32"),
                        tf.ones(batch_size, dtype="int32"),
                    ],
                    0,
                )

                for _ in range(c.D_STEPS):
                    indices = np.random.choice(x_concat_batch.shape[0], batch_size, replace=False)

                    self.D_step(
                        tf.gather(x_concat_batch, indices),
                        tf.gather(y_concat_batch, indices),
                    )

                b.add(1, values=[("loss(D)", self.D_loss.result())])

            self.history["pre_D_loss"].append(self.D_loss.result().numpy())

    def fit(
        self,
        batches: tf.data.Dataset,
        epochs: int = 10,
        g_epochs: int = 1,
        d_epochs: int = 5,
        n_rollouts: int = 16,
    ) -> None:
        """Train with Monte Carlo rewards and generator updates before discriminator updates.

        Rollouts complete each generated prefix using its original sampled start token as context.
        Reset loss metrics each epoch and append generator and discriminator losses to history.
        Only target sequences from each dataset element are used during adversarial training.

        Args:
            batches: Finite dataset of (input_ids, target_ids) batches with fixed batch and sequence dimensions.
            epochs: The maximum number of total training epochs.
            g_epochs: The maximum number of generator epochs per total epoch.
            d_epochs: The maximum number of discriminator epochs per total epoch.
            n_rollouts: Number of rollouts for conducting the Monte Carlo search.

        Raises:
            ValueError: Reward computation receives fewer than one rollout.

        """

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for _ in range(epochs):
            self.G_loss.reset_state()
            self.D_loss.reset_state()

            b = Progbar(n_batches, stateful_metrics=["loss(G)", "loss(D)"])

            for _, y_batch in batches:
                batch_size, max_length = y_batch.shape[0], y_batch.shape[1]

                for _ in range(g_epochs):
                    x_fake_batch, y_fake_batch = self.generate_batch(batch_size, max_length)

                    rewards = self._get_reward(y_fake_batch, n_rollouts, start_tokens=x_fake_batch[:, :1])

                    self.G_step(x_fake_batch, y_fake_batch, rewards)

                for _ in range(d_epochs):
                    _, y_fake_batch = self.generate_batch(batch_size, max_length)

                    x_concat_batch = tf.concat([y_batch, y_fake_batch], 0)
                    y_concat_batch = tf.concat(
                        [
                            tf.zeros(batch_size, dtype="int32"),
                            tf.ones(batch_size, dtype="int32"),
                        ],
                        0,
                    )

                    for _ in range(c.D_STEPS):
                        indices = np.random.choice(x_concat_batch.shape[0], batch_size, replace=False)

                        self.D_step(
                            tf.gather(x_concat_batch, indices),
                            tf.gather(y_concat_batch, indices),
                        )

                b.add(
                    1,
                    values=[
                        ("loss(G)", self.G_loss.result()),
                        ("loss(D)", self.D_loss.result()),
                    ],
                )

            self.history["G_loss"].append(self.G_loss.result().numpy())
            self.history["D_loss"].append(self.D_loss.result().numpy())
