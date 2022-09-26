"""Relational Generative Adversarial Network.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from nalp.core import Adversarial
from nalp.core.dataset import Dataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.discriminators import TextDiscriminator
from nalp.models.generators import GumbelRMCGenerator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class RelGAN(Adversarial):
    """A RelGAN class is the one in charge of Relational Generative Adversarial Networks implementation.

    References:
        W. Nie, N. Narodytska, A. Patel. Relgan: Relational generative adversarial networks for text generation.
        International Conference on Learning Representations (2018).

    """

    def __init__(
        self,
        encoder: Optional[IntegerEncoder] = None,
        vocab_size: Optional[int] = 1,
        max_length: Optional[int] = 1,
        embedding_size: Optional[int] = 32,
        n_slots: Optional[int] = 3,
        n_heads: Optional[int] = 5,
        head_size: Optional[int] = 10,
        n_blocks: Optional[int] = 1,
        n_layers: Optional[int] = 3,
        n_filters: Optional[Tuple[int, ...]] = (64),
        filters_size: Optional[Tuple[int, ...]] = (1),
        dropout_rate: Optional[float] = 0.25,
        tau: Optional[float] = 5.0,
    ):
        """Initialization method.

        Args:
            encoder: An index to vocabulary encoder for the generator.
            vocab_size: The size of the vocabulary for both discriminator and generator.
            max_length: Maximum length of the sequences for the discriminator.
            embedding_size: The size of the embedding layer for both discriminator and generator.
            n_slots: Number of memory slots for the generator.
            n_heads: Number of attention heads for the generator.
            head_size: Size of each attention head for the generator.
            n_blocks: Number of feed-forward networks for the generator.
            n_layers: Amout of layers per feed-forward network for the generator.
            n_filters: Number of filters to be applied in the discriminator.
            filters_size: Size of filters to be applied in the discriminator.
            dropout_rate: Dropout activation rate.
            tau: Gumbel-Softmax temperature parameter.

        """

        logger.info("Overriding class: Adversarial -> RelGAN.")

        D = TextDiscriminator(
            max_length, embedding_size, n_filters, filters_size, dropout_rate
        )
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

        super(RelGAN, self).__init__(D, G, name="RelGAN")

        self.vocab_size = vocab_size
        self.init_tau = tau

        logger.info("Class overrided.")

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary."""

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    @property
    def init_tau(self) -> float:
        """Gumbel-Softmax initial temperature."""

        return self._init_tau

    @init_tau.setter
    def init_tau(self, init_tau: float) -> None:
        self._init_tau = init_tau

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

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits
        self.D_loss = tf.metrics.Mean(name="D_loss")
        self.G_loss = tf.metrics.Mean(name="G_loss")

        self.history["pre_G_loss"] = []
        self.history["D_loss"] = []
        self.history["G_loss"] = []

    def generate_batch(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generates a batch of tokens by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            x: A tensor containing the inputs.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor]): A (batch_size, length) tensor of generated tokens and a
            (batch_size, length, vocab_size) tensor of predictions.

        """

        batch_size, max_length = x.shape[0], x.shape[1]

        start_batch = tf.expand_dims(x[:, 0], -1)

        sampled_preds = tf.zeros([batch_size, 0, self.vocab_size])
        sampled_batch = start_batch

        self.G.reset_states()

        for _ in range(max_length):
            _, preds, start_batch = self.G(start_batch)

            sampled_preds = tf.concat([sampled_preds, preds], 1)
            sampled_batch = tf.concat([sampled_batch, start_batch], 1)

        sampled_batch = sampled_batch[:, 1:]

        return sampled_batch, sampled_preds

    def _discriminator_loss(self, y_real: tf.Tensor, y_fake: tf.Tensor) -> tf.Tensor:
        """Calculates the loss out of the discriminator architecture.

        Args:
            y_real: A tensor containing the real data targets.
            y_fake: A tensor containing the fake data targets.

        Returns:
            (tf.Tensor): The loss based on the discriminator network.

        """

        loss = self.loss(tf.ones_like(y_real), y_real - y_fake)

        return tf.reduce_mean(loss)

    def _generator_loss(self, y_real: tf.Tensor, y_fake: tf.Tensor) -> tf.Tensor:
        """Calculates the loss out of the generator architecture.

        Args:
            y_real: A tensor containing the real data targets.
            y_fake: A tensor containing the fake data targets.

        Returns:
            (tf.Tensor): The loss based on the generator network.

        """

        loss = self.loss(tf.ones_like(y_fake), y_fake - y_real)

        return tf.reduce_mean(loss)

    @tf.function
    def G_pre_step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            x: A tensor containing the inputs.
            y: A tensor containing the inputs' labels.

        """

        with tf.GradientTape() as tape:
            logits, _, _ = self.G(x)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)
            )

        gradients = tape.gradient(loss, self.G.trainable_variables)

        self.P_optimizer.apply_gradients(zip(gradients, self.G.trainable_variables))

        self.G_loss.update_state(loss)

    @tf.function
    def step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Performs a single batch optimization step.

        Args:
            x: A tensor containing the inputs.
            y: A tensor containing the inputs' labels.

        """

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # Generates new data, e.g., G(x)
            _, x_fake_probs = self.generate_batch(x)

            # Samples fake targets from D(G(x))
            y_fake = self.D(x_fake_probs)

            # Extends the target tensor to an one-hot encoding representation
            # and samples real targets from D(x)
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

    def pre_fit(self, batches: Dataset, epochs: Optional[int] = 100) -> None:
        """Pre-trains the model.

        Args:
            batches: Pre-training batches containing samples.
            epochs: The maximum number of pre-training epochs.

        """

        logger.info("Pre-fitting generator ...")

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for e in range(epochs):
            logger.info("Epoch %d/%d", e + 1, epochs)

            self.G_loss.reset_states()

            b = Progbar(n_batches, stateful_metrics=["loss(G)"])

            for x_batch, y_batch in batches:
                self.G_pre_step(x_batch, y_batch)

                b.add(1, values=[("loss(G)", self.G_loss.result())])

            self.history["pre_G_loss"].append(self.G_loss.result().numpy())

            logger.to_file("Loss(G): %s", self.G_loss.result().numpy())

    def fit(self, batches: Dataset, epochs: Optional[int] = 100) -> None:
        """Trains the model.

        Args:
            batches: Training batches containing samples.
            epochs: The maximum number of training epochs.

        """

        logger.info("Fitting model ...")

        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for e in range(epochs):
            logger.info("Epoch %d/%d", e + 1, epochs)

            self.G_loss.reset_states()
            self.D_loss.reset_states()

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

            # Exponentially annealing the Gumbel-Softmax temperature
            self.G.tau = self.init_tau ** ((epochs - e) / epochs)

            self.history["G_loss"].append(self.G_loss.result().numpy())
            self.history["D_loss"].append(self.D_loss.result().numpy())

            logger.to_file(
                "Loss(G): %s | Loss(D): %s",
                self.G_loss.result().numpy(),
                self.D_loss.result().numpy(),
            )
