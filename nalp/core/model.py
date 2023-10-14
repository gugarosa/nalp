"""Model-related classes.
"""

from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar

import nalp.utils.constants as c
from nalp.core.dataset import Dataset
from nalp.utils import logging

logger = logging.get_logger(__name__)


class Discriminator(Model):
    """A Discriminator class is responsible for easily-implementing the discriminative part of
    a neural network, when custom training or additional sets are not needed.

    """

    def __init__(self, name: str = "") -> None:
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name: The model's identifier string.

        """

        super(Discriminator, self).__init__(name=name)

    def call(self, x: tf.Tensor, training: bool = True) -> None:
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x: A tensorflow's tensor holding input data.
            training: Whether architecture is under training or not.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError


class Generator(Model):
    """A Generator class is responsible for easily-implementing the generative part of
    a neural network, when custom training or additional sets are not needed.

    """

    def __init__(self, name: str = "") -> None:
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name: The model's identifier string.

        """

        super(Generator, self).__init__(name=name)

    def call(self, x: tf.Tensor, training: bool = True) -> None:
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x: A tensorflow's tensor holding input data.
            training: Whether architecture is under training or not.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def generate_greedy_search(
        self, start: str, max_length: int = 100
    ) -> List[str]:
        """Generates text by using greedy search, where the sampled
        token is always sampled according to the maximum probability.

        Args:
            start: The start string to generate the text.
            max_length: Maximum length of generated text.

        Returns:
            (List[str]): Generated text.

        """

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_states()

        sampled_tokens = []
        for _ in range(max_length):
            preds = self(start_tokens)
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
        start: str,
        max_length: int = 100,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generates text by using temperature sampling, where the sampled
        token is sampled according to a multinomial/categorical distribution.

        Args:
            start: The start string to generate the text.
            max_length: Length of generated text.
            temperature: A temperature value to sample the token.

        Returns:
            (List[str]): Generated text.

        """

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_states()

        sampled_tokens = []
        for _ in range(max_length):
            preds = self(start_tokens)
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
        start: str,
        max_length: int = 100,
        k: int = 0,
        p: float = 0.0,
    ) -> List[str]:
        """Generates text by using top-k and top-p sampling, where the sampled
        token is sampled according to the `k` most likely words distribution, as well
        as to the maximum cumulative probability `p`.

        Args:
            start: The start string to generate the text.
            max_length: Length of generated text.
            k: Indicates the amount of likely words.
            p: Maximum cumulative probability to be thresholded.

        Returns:
            (List[str]): Generated text.

        """

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_states()

        sampled_tokens = []
        for _ in range(max_length):
            preds = self(start_tokens)
            preds = preds[:, -1, :]

            if k > 0:
                preds, preds_indexes = tf.math.top_k(preds, k)
            else:
                preds, preds_indexes = tf.math.top_k(preds, preds.shape[-1])

            if p > 0.0:
                cum_probs = tf.math.cumsum(tf.nn.softmax(preds), axis=-1)

                # Also ensures that first index will always be true to prevent zero
                # tokens from being sampled
                ignored_indexes = cum_probs <= p
                ignored_indexes = tf.tensor_scatter_nd_update(
                    ignored_indexes, [[0, 0]], [True]
                )

                preds = tf.expand_dims(preds[ignored_indexes], 0)
                preds_indexes = tf.expand_dims(preds_indexes[ignored_indexes], 0)

            # Samples the maximum top-k logit and gathers the real token index
            index = tf.random.categorical(preds, 1)[0, 0]
            sampled_token = [preds_indexes[-1][index].numpy()]

            start_tokens = tf.expand_dims(sampled_token, 0)

            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            if sampled_token == c.EOS:
                break

        return sampled_tokens


class Adversarial(Model):
    """An Adversarial class is responsible for customly
    implementing Generative Adversarial Networks.

    """

    def __init__(
        self,
        discriminator: Discriminator,
        generator: Generator,
        name: str = "",
    ) -> None:
        """Initialization method.

        Args:
            discriminator: Network's discriminator architecture.
            generator: Network's generator architecture.
            name: The model's identifier string.

        """

        super(Adversarial, self).__init__(name=name)

        self.D = discriminator
        self.G = generator
        self.history = {}

    @property
    def D(self) -> Discriminator:
        """Discriminator architecture."""

        return self._D

    @D.setter
    def D(self, D: Discriminator) -> None:
        self._D = D

    @property
    def G(self) -> Generator:
        """Generator architecture."""

        return self._G

    @G.setter
    def G(self, G: Generator) -> None:
        self._G = G

    @property
    def history(self) -> Dict[str, Any]:
        """History dictionary."""

        return self._history

    @history.setter
    def history(self, history: Dict[str, Any]) -> None:
        self._history = history

    def compile(
        self, d_optimizer: tf.keras.optimizers, g_optimizer: tf.keras.optimizers
    ) -> None:
        """Main building method.

        Args:
            d_optimizer: An optimizer instance for the discriminator.
            g_optimizer: An optimizer instance for the generator.

        """

        self.D_optimizer = d_optimizer
        self.G_optimizer = g_optimizer

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits
        self.D_loss = tf.metrics.Mean(name="D_loss")
        self.G_loss = tf.metrics.Mean(name="G_loss")

        self.history["D_loss"] = []
        self.history["G_loss"] = []

    def _discriminator_loss(self, y_real: tf.Tensor, y_fake: tf.Tensor) -> tf.Tensor:
        """Calculates the loss out of the discriminator architecture.

        Args:
            y_real: A tensor containing the real data targets.
            y_fake: A tensor containing the fake data targets.

        Returns:
            (tf.Tensor): The loss based on the discriminator network.

        """

        real_loss = self.loss(tf.ones_like(y_real), y_real)
        fake_loss = self.loss(tf.zeros_like(y_fake), y_fake)

        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    def _generator_loss(self, y_fake: tf.Tensor) -> tf.Tensor:
        """Calculates the loss out of the generator architecture.

        Args:
            y_fake: A tensor containing the fake data targets.

        Returns:
            (tf.Tensor): The loss based on the generator network.

        """

        loss = self.loss(tf.ones_like(y_fake), y_fake)

        return tf.reduce_mean(loss)

    @tf.function
    def step(self, x: tf.Tensor) -> None:
        """Performs a single batch optimization step.

        Args:
            x: A tensor containing the inputs.

        """

        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # Samples fake targets D(G(z)) and real targets D(x) from the discriminator
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

    def fit(self, batches: Dataset, epochs: int = 100) -> None:
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

            logger.to_file(
                "Loss(G): %s | Loss(D): %s",
                self.G_loss.result().numpy(),
                self.D_loss.result().numpy(),
            )
