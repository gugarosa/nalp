# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Wasserstein Generative Adversarial Network."""

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from nalp.core.model import Adversarial
from nalp.models.discriminators.conv import ConvDiscriminator
from nalp.models.generators.conv import ConvGenerator


class WGAN(Adversarial):
    """Train a Wasserstein GAN with weight clipping or gradient penalty."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (28, 28, 1),
        noise_dim: int = 100,
        n_samplings: int = 3,
        alpha: float = 0.3,
        dropout_rate: float = 0.3,
        model_type: str = "wc",
        clip: float = 0.01,
        penalty: int = 10,
    ) -> None:
        """Initialize the convolutional generator and Wasserstein critic.

        Weight clipping constrains critic parameters after each update. Gradient penalty instead penalizes
        gradient norms on interpolated real and generated images.

        References: M. Arjovsky, S. Chintala, L. Bottou. Wasserstein gan. Preprint arXiv:1701.07875 (2017).
        I. Gulrajani, et al. Improved training of wasserstein gans.
        Advances in neural information processing systems (2017).

        Args:
            input_shape: Target image shape in height, width, and channels.
            noise_dim: Number of noise dimensions for the generator.
            n_samplings: Number of discriminator downsamplings and generator upsamplings.
            alpha: Negative slope of the LeakyReLU activation.
            dropout_rate: Dropout activation rate.
            model_type: Algorithm selector for weight clipping ("wc") or gradient penalty ("gp").
            clip: Symmetric critic weight bound for the Lipschitz constraint.
            penalty: Coefficient for the gradient penalty.

        """

        D = ConvDiscriminator(n_samplings, alpha, dropout_rate)
        G = ConvGenerator(input_shape, noise_dim, n_samplings, alpha)

        super().__init__(D, G, name="wgan")

        self.model_type = model_type
        self.clip = clip
        self.penalty_lambda = penalty

    def _gradient_penalty(self, x: tf.Tensor, x_fake: tf.Tensor) -> tf.Tensor:
        e = tf.random.uniform([x.shape[0], 1, 1, 1])

        x_penalty = x * e + (1 - e) * x_fake
        with tf.GradientTape() as tape:
            tape.watch(x_penalty)

            y_penalty = self.D(x_penalty)

        penalty_gradients = tape.gradient(y_penalty, x_penalty)
        penalty_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(penalty_gradients), [1, 2, 3]))

        penalty = tf.reduce_mean((penalty_gradients_norm - 1) ** 2)

        return penalty

    @tf.function
    def D_step(self, x: tf.Tensor) -> None:
        """Update the critic once and accumulate its Wasserstein loss.

        Apply gradient penalty inside the loss or clip weights after the update, according to model_type.

        Args:
            x: Real channels-last image tensor with a fixed leading batch dimension.

        """

        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])
        with tf.GradientTape() as tape:
            x_fake = self.G(z)

            y_fake = self.D(x_fake)
            y_real = self.D(x)

            D_loss = -tf.reduce_mean(y_real) + tf.reduce_mean(y_fake)

            if self.model_type == "gp":
                penalty = self._gradient_penalty(x, x_fake)
                D_loss += penalty * self.penalty_lambda

        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)

        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))

        self.D_loss.update_state(D_loss)

        if self.model_type == "wc":
            [w.assign(tf.clip_by_value(w, -self.clip, self.clip)) for w in self.D.trainable_variables]

    @tf.function
    def G_step(self, x: tf.Tensor) -> None:
        """Update the generator once and accumulate its Wasserstein loss.

        Args:
            x: Real image tensor whose leading dimension determines the generated batch size.

        """

        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])
        with tf.GradientTape() as tape:
            x_fake = self.G(z)

            y_fake = self.D(x_fake)

            G_loss = -tf.reduce_mean(y_fake)

        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)

        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))

        self.G_loss.update_state(G_loss)

    def fit(
        self,
        batches: tf.data.Dataset,
        epochs: int = 100,
        critic_steps: int = 5,
    ) -> None:
        """Train the critic repeatedly before each generator update.

        Reset loss metrics each epoch and append generator and discriminator losses to history.

        Args:
            batches: Finite dataset of channels-last image batches without labels and with fixed batch dimensions.
            epochs: The maximum number of training epochs.
            critic_steps: Number of critic updates before each generator update.

        """

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for _ in range(epochs):
            self.G_loss.reset_state()
            self.D_loss.reset_state()

            b = Progbar(n_batches, stateful_metrics=["loss(G)", "loss(D)"])

            for batch in batches:
                for _ in range(critic_steps):
                    self.D_step(batch)

                self.G_step(batch)

                b.add(
                    1,
                    values=[
                        ("loss(G)", self.G_loss.result()),
                        ("loss(D)", self.D_loss.result()),
                    ],
                )

            self.history["G_loss"].append(self.G_loss.result().numpy())
            self.history["D_loss"].append(self.D_loss.result().numpy())
