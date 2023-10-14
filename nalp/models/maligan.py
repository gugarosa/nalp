"""Maximum-Likelihood Augmented Discrete Generative Adversarial Network.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

import nalp.utils.constants as c
from nalp.core import Adversarial
from nalp.core.dataset import Dataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.discriminators import EmbeddedTextDiscriminator
from nalp.models.generators import LSTMGenerator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class MaliGAN(Adversarial):
    """A MaliGAN class is the one in charge of Maximum-Likelihood Augmented Discrete
    Generative Adversarial Networks implementation.

    References:
        T. Che, et al. Maximum-likelihood augmented discrete generative adversarial networks.
        Preprint arXiv:1702.07983 (2017).

    """

    def __init__(
        self,
        encoder: Optional[IntegerEncoder] = None,
        vocab_size: int = 1,
        max_length: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        n_filters: Tuple[int, ...] = (64),
        filters_size: Tuple[int, ...] = (1),
        dropout_rate: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        """Initialization method.

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

        logger.info("Overriding class: Adversarial -> MaliGAN.")

        D = EmbeddedTextDiscriminator(
            vocab_size,
            max_length,
            embedding_size,
            n_filters,
            filters_size,
            dropout_rate,
        )
        G = LSTMGenerator(encoder, vocab_size, embedding_size, hidden_size)

        super(MaliGAN, self).__init__(D, G, name="maligan")

        self.vocab_size = vocab_size
        self.T = temperature

        logger.info("Class overrided.")

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary."""

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    @property
    def T(self) -> float:
        """Temperature value to sample the token."""

        return self._T

    @T.setter
    def T(self, T: float) -> None:
        self._T = T

    def compile(
        self,
        pre_optimizer: tf.keras.optimizers,
        d_optimizer: tf.keras.optimizers,
        g_optimizer: tf.keras.optimizers,
    ) -> None:
        """Main building method.

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

    def generate_batch(
        self, batch_size: int = 1, length: int = 1
    ) -> tf.Tensor:
        """Generates a batch of tokens by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            batch_size: Size of the batch to be generated.
            length: Length of generated tokens.

        Returns:
            (tf.Tensor): A (batch_size, length) tensor of generated tokens.

        """

        start_batch = tf.random.uniform(
            [batch_size, 1], 0, self.vocab_size, dtype="int32"
        )
        sampled_batch = start_batch

        self.G.reset_states()

        for _ in range(length):
            preds = self.G(start_batch)

            preds = tf.squeeze(preds, 1)
            preds /= self.T

            start_batch = tf.random.categorical(preds, 1, dtype="int32")
            sampled_batch = tf.concat([sampled_batch, start_batch], 1)

        x_sampled_batch = sampled_batch[:, :length]
        y_sampled_batch = sampled_batch[:, 1:]

        return x_sampled_batch, y_sampled_batch

    def _get_reward(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates rewards over an input using a Maximum-Likelihood approach.

        Args:
            x: A tensor containing the inputs.

        Returns:
            (tf.Tensor): Reward over input.

        """

        batch_size, max_length = x.shape[0], x.shape[1]

        rewards = tf.squeeze(self.D(x), 1)[:, 1]
        rewards = tf.math.divide(rewards, 1 - rewards)
        rewards = tf.math.divide(rewards, tf.math.reduce_sum(rewards))
        rewards = tf.broadcast_to(tf.expand_dims(rewards, 1), [batch_size, max_length])

        return rewards

    @tf.function
    def G_pre_step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            x: A tensor containing the inputs.
            y: A tensor containing the inputs' labels.

        """

        with tf.GradientTape() as tape:
            preds = self.G(x)

            loss = tf.reduce_mean(self.loss(y, preds))

        gradients = tape.gradient(loss, self.G.trainable_variables)

        self.P_optimizer.apply_gradients(zip(gradients, self.G.trainable_variables))

        self.G_loss.update_state(loss)

    @tf.function
    def G_step(self, x: tf.Tensor, y: tf.Tensor, rewards: tf.Tensor) -> None:
        """Performs a single batch optimization step over the generator.

        Args:
            x : A tensor containing the inputs.
            y: A tensor containing the inputs' labels.
            rewards: A tensor containing the rewards for the input.

        """

        with tf.GradientTape() as tape:
            preds = self.G(x)

            loss = tf.reduce_mean(self.loss(y, preds) * rewards)

        gradients = tape.gradient(loss, self.G.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients, self.G.trainable_variables))

        self.G_loss.update_state(loss)

    @tf.function
    def D_step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Performs a single batch optimization step over the discriminator.

        Args:
            x: A tensor containing the inputs.
            y: A tensor containing the inputs' labels.

        """

        with tf.GradientTape() as tape:
            preds = tf.squeeze(self.D(x), 1)

            loss = tf.reduce_mean(self.loss(y, preds))

        gradients = tape.gradient(loss, self.D.trainable_variables)

        self.D_optimizer.apply_gradients(zip(gradients, self.D.trainable_variables))

        self.D_loss.update_state(loss)

    def pre_fit(
        self,
        batches: Dataset,
        g_epochs: int = 50,
        d_epochs: int = 10,
    ) -> None:
        """Pre-trains the model.

        Args:
            batches: Pre-training batches containing samples.
            g_epochs: The maximum number of pre-training generator epochs.
            d_epochs: The maximum number of pre-training discriminator epochs.

        """

        logger.info("Pre-fitting generator ...")

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for e in range(g_epochs):
            logger.info("Epoch %d/%d", e + 1, g_epochs)

            self.G_loss.reset_states()

            b = Progbar(n_batches, stateful_metrics=["loss(G)"])

            for x_batch, y_batch in batches:
                self.G_pre_step(x_batch, y_batch)

                b.add(1, values=[("loss(G)", self.G_loss.result())])

            self.history["pre_G_loss"].append(self.G_loss.result().numpy())

            logger.to_file("Loss(G): %s", self.G_loss.result().numpy())

        logger.info("Pre-fitting discriminator ...")

        for e in range(d_epochs):
            logger.info("Epoch %d/%d", e + 1, d_epochs)

            self.D_loss.reset_states()

            b = Progbar(n_batches, stateful_metrics=["loss(D)"])

            for x_batch, _ in batches:
                batch_size, max_length = x_batch.shape[0], x_batch.shape[1]

                x_fake_batch, _ = self.generate_batch(batch_size, max_length)

                x_concat_batch = tf.concat([x_batch, x_fake_batch], 0)
                y_concat_batch = tf.concat(
                    [
                        tf.zeros(batch_size, dtype="int32"),
                        tf.ones(batch_size, dtype="int32"),
                    ],
                    0,
                )

                for _ in range(c.D_STEPS):
                    indices = np.random.choice(
                        x_concat_batch.shape[0], batch_size, replace=False
                    )

                    self.D_step(
                        tf.gather(x_concat_batch, indices),
                        tf.gather(y_concat_batch, indices),
                    )

                b.add(1, values=[("loss(D)", self.D_loss.result())])

            self.history["pre_D_loss"].append(self.D_loss.result().numpy())

            logger.to_file("Loss(D): %s", self.D_loss.result().numpy())

    def fit(
        self, batches: Dataset, epochs: int = 10, d_epochs: int = 5
    ) -> None:
        """Trains the model.

        Args:
            batches: Training batches containing samples.
            epochs: The maximum number of total training epochs.
            d_epochs: The maximum number of discriminator epochs per total epoch.

        """

        logger.info("Fitting model ...")

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for e in range(epochs):
            logger.info("Epoch %d/%d", e + 1, epochs)

            self.G_loss.reset_states()
            self.D_loss.reset_states()

            b = Progbar(n_batches, stateful_metrics=["loss(G)", "loss(D)"])

            for x_batch, _ in batches:
                batch_size, max_length = x_batch.shape[0], x_batch.shape[1]

                for _ in range(d_epochs):
                    x_fake_batch, _ = self.generate_batch(batch_size, max_length)

                    x_concat_batch = tf.concat([x_batch, x_fake_batch], 0)
                    y_concat_batch = tf.concat(
                        [
                            tf.zeros(batch_size, dtype="int32"),
                            tf.ones(batch_size, dtype="int32"),
                        ],
                        0,
                    )

                    for _ in range(c.D_STEPS):
                        indices = np.random.choice(
                            x_concat_batch.shape[0], batch_size, replace=False
                        )

                        self.D_step(
                            tf.gather(x_concat_batch, indices),
                            tf.gather(y_concat_batch, indices),
                        )

                x_fake_batch, y_fake_batch = self.generate_batch(batch_size, max_length)
                rewards = self._get_reward(x_fake_batch)

                self.G_step(x_fake_batch, y_fake_batch, rewards)

                b.add(
                    1,
                    values=[
                        ("loss(G)", self.G_loss.result()),
                        ("loss(D)", self.D_loss.result()),
                    ],
                )

            self.history["G_loss"].append(self.G_loss.result().numpy())
            self.history["D_loss"].append(self.D_loss.result().numpy())

            logger.to_file(
                "Loss(G): %s | Loss(D): %s",
                self.G_loss.result().numpy(),
                self.D_loss.result().numpy(),
            )
