# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Model-related classes."""

import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar

import nalp.utils.constants as c


class Discriminator(Model):
    """Define the discriminator component implemented by concrete model families."""

    def __init__(self, name: str = "") -> None:
        """Initialize a named discriminator.

        Args:
            name: Keras model identifier.

        """

        super().__init__(name=name)

    def call(self, x: tf.Tensor, training: bool = True) -> None:
        raise NotImplementedError("`Discriminator.call` must be implemented by a subclass.")


class Generator(Model):
    """Define a generator component and shared autoregressive token sampling."""

    def __init__(self, name: str = "") -> None:
        """Initialize a named generator.

        Args:
            name: Keras model identifier.

        """

        super().__init__(name=name)

    def call(self, x: tf.Tensor, training: bool = True) -> None:
        raise NotImplementedError("`Generator.call` must be implemented by a subclass.")

    def reset_state(self) -> None:
        """Reset built recurrent layers before starting an independent sequence.

        Unbuilt layers are left unchanged.

        """

        for layer in self.layers:
            states = getattr(layer, "states", None)
            if states is None:
                continue

            for state in tf.nest.flatten(states):
                state.assign(tf.zeros_like(state))

    def reset_states(self) -> None:
        """Reset stateful recurrent layers."""

        self.reset_state()

    def _generation_logits(self, x: tf.Tensor) -> tf.Tensor:
        return self(x)

    def generate_greedy_search(self, start: str | list[str], max_length: int = 100) -> list[str]:
        """Generate tokens by selecting the largest next-token logit.

        Reset recurrent state before sampling and use the attached learned encoder to translate tokens and IDs.

        Args:
            start: Character string or pre-tokenized prompt matching the encoder.
            max_length: Maximum number of generated tokens without counting the prompt.

        Returns:
            Decoded sampled tokens, including an end-of-sentence marker when one is generated.

        Raises:
            RuntimeError: The attached encoder has not learned a mapping.

        """

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_state()

        sampled_tokens = []
        for _ in range(max_length):
            preds = self._generation_logits(start_tokens)
            preds = preds[:, -1, :]

            sampled_token = tf.argmax(preds, 1).numpy()

            start_tokens = tf.expand_dims(sampled_token, 0)

            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_temperature_sampling(
        self,
        start: str | list[str],
        max_length: int = 100,
        temperature: float = 1.0,
    ) -> list[str]:
        """Sample next-token logits from a temperature-scaled categorical distribution.

        Reset recurrent state before sampling and stop on an end-of-sentence token or the length limit.

        Args:
            start: Character string or pre-tokenized prompt matching the encoder.
            max_length: Maximum number of generated tokens without counting the prompt.
            temperature: Finite positive divisor applied to logits before categorical sampling.

        Returns:
            Decoded sampled tokens, including an end-of-sentence marker when one is generated.

        Raises:
            ValueError: The temperature is non-finite or not positive.
            RuntimeError: The attached encoder has not learned a mapping.

        """

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"`temperature` must be finite and positive, but got {temperature}.")

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_state()

        sampled_tokens = []
        for _ in range(max_length):
            preds = self._generation_logits(start_tokens)
            preds = preds[:, -1, :]

            preds /= temperature

            sampled_token = tf.random.categorical(preds, 1)[0].numpy()

            start_tokens = tf.expand_dims(sampled_token, 0)

            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_top_sampling(
        self,
        start: str | list[str],
        max_length: int = 100,
        k: int = 0,
        p: float = 0.0,
    ) -> list[str]:
        """Sample tokens from top-k logits and an optional nucleus probability prefix.

        Reset recurrent state before sampling and apply nucleus filtering to the retained top-k distribution.

        Args:
            start: Character string or pre-tokenized prompt matching the encoder.
            max_length: Maximum number of generated tokens without counting the prompt.
            k: Number of highest-scoring tokens retained, with 0 retaining the complete vocabulary.
            p: Cumulative probability threshold after top-k selection, with 0 disabling nucleus filtering.

        Returns:
            Decoded sampled tokens, including an end-of-sentence marker when one is generated.

        Raises:
            ValueError: The top-k count is negative or the probability threshold is outside the closed unit interval.
            RuntimeError: The attached encoder has not learned a mapping.

        """

        if k < 0:
            raise ValueError(f"`k` must be nonnegative, but got {k}.")
        if not 0 <= p <= 1:
            raise ValueError(f"`p` must be between 0 and 1, but got {p}.")

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_state()

        sampled_tokens = []
        for _ in range(max_length):
            preds = self._generation_logits(start_tokens)
            preds = preds[:, -1, :]

            if k > 0:
                preds, preds_indexes = tf.math.top_k(preds, k)
            else:
                preds, preds_indexes = tf.math.top_k(preds, preds.shape[-1])

            if p > 0.0:
                cumulative_before = tf.math.cumsum(tf.nn.softmax(preds), axis=-1, exclusive=True)
                keep = cumulative_before < p

                preds = tf.expand_dims(preds[keep], 0)
                preds_indexes = tf.expand_dims(preds_indexes[keep], 0)

            index = tf.random.categorical(preds, 1)[0, 0]
            sampled_token = [preds_indexes[-1][index].numpy()]

            start_tokens = tf.expand_dims(sampled_token, 0)

            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            if sampled_token == c.EOS:
                break

        return sampled_tokens


class Adversarial(Model):
    """Coordinate generator and discriminator updates with explicit training loops."""

    def __init__(
        self,
        discriminator: Discriminator,
        generator: Generator,
        name: str = "",
    ) -> None:
        """Initialize a trainer around existing discriminator and generator models.

        Retain both models by reference and create the history container populated by training.

        Args:
            discriminator: Discriminator component updated during training.
            generator: Generator component updated during training.
            name: Keras model identifier.

        """

        super().__init__(name=name)

        self.D = discriminator
        self.G = generator
        self.history: dict[str, Any] = {}

    def compile(self, d_optimizer: tf.keras.optimizers.Optimizer, g_optimizer: tf.keras.optimizers.Optimizer) -> None:
        """Configure optimizers and reset adversarial loss tracking.

        Store the optimizer instances, create fresh loss metrics, and reset the D_loss and G_loss history series.

        Args:
            d_optimizer: Optimizer instance for discriminator variables.
            g_optimizer: Optimizer instance for generator variables.

        """

        self.D_optimizer = d_optimizer
        self.G_optimizer = g_optimizer

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits
        self.D_loss = tf.metrics.Mean(name="D_loss")
        self.G_loss = tf.metrics.Mean(name="G_loss")

        self.history["D_loss"] = []
        self.history["G_loss"] = []

    def _discriminator_loss(self, y_real: tf.Tensor, y_fake: tf.Tensor) -> tf.Tensor:
        real_loss = self.loss(tf.ones_like(y_real), y_real)
        fake_loss = self.loss(tf.zeros_like(y_fake), y_fake)

        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    def _generator_loss(self, y_fake: tf.Tensor) -> tf.Tensor:
        loss = self.loss(tf.ones_like(y_fake), y_fake)

        return tf.reduce_mean(loss)

    @tf.function
    def step(self, x: tf.Tensor) -> None:
        """Apply one generator update and one discriminator update for an image batch.

        Accumulate both losses in the current metrics.

        Args:
            x: Real image samples with a fixed leading batch dimension.

        """

        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            x_fake = self.G(z)

            y_fake = self.D(x_fake)
            y_real = self.D(x)

            G_loss = self._generator_loss(y_fake)
            D_loss = self._discriminator_loss(y_real, y_fake)

        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))

        self.G_loss.update_state(G_loss)
        self.D_loss.update_state(D_loss)

    def fit(self, batches: tf.data.Dataset, epochs: int = 100) -> None:
        """Train both components and append epoch-mean losses to history.

        Use NALP's explicit training loop rather than the full Keras fit interface.

        Args:
            batches: TensorFlow dataset yielding real image batches rather than a NALP dataset wrapper.
            epochs: Number of complete passes over the batches.

        """

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for _ in range(epochs):
            self.G_loss.reset_state()
            self.D_loss.reset_state()

            b = Progbar(n_batches, stateful_metrics=["loss(G)", "loss(D)"])

            for batch in batches:
                self.step(batch)

                b.add(
                    1,
                    values=[
                        ("loss(G)", self.G_loss.result()),
                        ("loss(D)", self.D_loss.result()),
                    ],
                )

            self.history["G_loss"].append(self.G_loss.result().numpy())
            self.history["D_loss"].append(self.D_loss.result().numpy())
