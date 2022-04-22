"""Wasserstein Generative Adversarial Network.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from nalp.core import Adversarial
from nalp.core.dataset import Dataset
from nalp.models.discriminators import ConvDiscriminator
from nalp.models.generators import ConvGenerator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class WGAN(Adversarial):
    """A WGAN class is the one in charge of Wasserstein Generative Adversarial Networks
    implementation with weight clipping or gradient penalty algorithms.

    References:
        M. Arjovsky, S. Chintala, L. Bottou.
        Wasserstein gan.
        Preprint arXiv:1701.07875 (2017).

        I. Gulrajani, et al.
        Improved training of wasserstein gans.
        Advances in neural information processing systems (2017).

    """

    def __init__(
        self,
        input_shape: Optional[Tuple[int, int, int]] = (28, 28, 1),
        noise_dim: Optional[int] = 100,
        n_samplings: Optional[int] = 3,
        alpha: Optional[float] = 0.3,
        dropout_rate: Optional[float] = 0.3,
        model_type: Optional[str] = "wc",
        clip: Optional[float] = 0.01,
        penalty: Optional[int] = 10,
    ):
        """Initialization method.

        Args:
            input_shape: An input shape for the Generator.
            noise_dim: Amount of noise dimensions for the Generator.
            n_samplings: Number of down/up samplings to perform.
            alpha: LeakyReLU activation threshold.
            dropout_rate: Dropout activation rate.
            model_type: Whether should use weight clipping (wc) or gradient penalty (gp).
            clip: Clipping value for the Lipschitz constrain.
            penalty: Coefficient for the gradient penalty.

        """

        logger.info("Overriding class: Adversarial -> WGAN.")

        # Creating the discriminator network
        D = ConvDiscriminator(n_samplings, alpha, dropout_rate)

        # Creating the generator network
        G = ConvGenerator(input_shape, noise_dim, n_samplings, alpha)

        super(WGAN, self).__init__(D, G, name="wgan")

        # Defining the type of penalization to be used
        self.model_type = model_type

        # Defining the clipping value as a property for further usage
        self.clip = clip

        # Defining the gradient penalty coefficient as a property for further usage
        self.penalty_lambda = penalty

        logger.debug(
            "Input: %s | Noise: %d | Number of samplings: %d | "
            "Activation rate: %s | Dropout rate: %s | Type: %s.",
            input_shape,
            noise_dim,
            n_samplings,
            alpha,
            dropout_rate,
            model_type,
        )

        logger.info("Class overrided.")

    @property
    def model_type(self) -> str:
        """Whether should use weight clipping (wc) or gradient penalty (gp)."""

        return self._model_type

    @model_type.setter
    def model_type(self, model_type: str) -> None:
        self._model_type = model_type

    @property
    def clip(self) -> float:
        """Clipping value for the Lipschitz constrain."""

        return self._clip

    @clip.setter
    def clip(self, clip: float) -> None:
        self._clip = clip

    @property
    def penalty_lambda(self) -> int:
        """Coefficient for the gradient penalty."""

        return self._penalty_lambda

    @penalty_lambda.setter
    def penalty_lambda(self, penalty_lambda: int) -> None:
        self._penalty_lambda = penalty_lambda

    def _gradient_penalty(self, x: tf.Tensor, x_fake: tf.Tensor) -> tf.Tensor:
        """Performs the gradient penalty procedure.

        Args:
            x: A tensor containing the real inputs.
            x_fake: A tensor containing the fake inputs.

        Returns:
            (tf.Tensor): The penalization to be applied over the loss function.

        """

        # Samples an uniform random number
        e = tf.random.uniform([x.shape[0], 1, 1, 1])

        # Calculates the penalized input
        x_penalty = x * e + (1 - e) * x_fake

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Watches the penalized input
            tape.watch(x_penalty)

            # Samples the score from the penalized input
            y_penalty = self.D(x_penalty)

        # Calculates the penalization gradients
        penalty_gradients = tape.gradient(y_penalty, x_penalty)

        # Calculates the norm of the penalization gradients
        penalty_gradients_norm = tf.sqrt(
            tf.reduce_sum(tf.square(penalty_gradients), [1, 2, 3])
        )

        # Calculates the gradient penalty
        penalty = tf.reduce_mean((penalty_gradients_norm - 1) ** 2)

        return penalty

    @tf.function
    def D_step(self, x: tf.Tensor) -> None:
        """Performs a single batch optimization step over the discriminator.

        Args:
            x: A tensor containing the inputs.

        """

        # Defines a random noise signal as the generator's input
        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # Samples fake scores from D(G(z)) and real scores from D(x)
            y_fake = self.D(x_fake)
            y_real = self.D(x)

            # Calculates the discriminator loss upon D(x) and D(G(z))
            D_loss = -tf.reduce_mean(y_real) + tf.reduce_mean(y_fake)

            # Checks if WGAN is using gradient penalty
            if self.model_type == "gp":
                # Calculates the penalization score
                penalty = self._gradient_penalty(x, x_fake)

                # Sums the gradient penalty over the discriminator loss
                D_loss += penalty * self.penalty_lambda

        # Calculate the gradients based on discriminator's loss for each training variable
        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)

        # Applies the discriminator's gradients using an optimizer
        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))

        # Updates the discriminator's loss state
        self.D_loss.update_state(D_loss)

        # Checks if WGAN is using weight clipping
        if self.model_type == "wc":
            # Clips the weights
            [
                w.assign(tf.clip_by_value(w, -self.clip, self.clip))
                for w in self.D.trainable_variables
            ]

    @tf.function
    def G_step(self, x: tf.Tensor) -> None:
        """Performs a single batch optimization step over the generator.

        Args:
            x: A tensor containing the inputs.

        """

        # Defines a random noise signal as the generator's input
        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # Samples fake targets from the discriminator, e.g., D(G(z))
            y_fake = self.D(x_fake)

            # Calculates the generator loss upon D(G(z))
            G_loss = -tf.reduce_mean(y_fake)

        # Calculate the gradients based on generator's loss for each training variable
        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)

        # Applies the generator's gradients using an optimizer
        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(G_loss)

    def fit(
        self,
        batches: Dataset,
        epochs: Optional[int] = 100,
        critic_steps: Optional[int] = 5,
    ) -> None:
        """Trains the model.

        Args:
            batches: Training batches containing samples.
            epochs: The maximum number of training epochs.
            critic_steps: Amount of discriminator epochs per training epoch.

        """

        logger.info("Fitting model ...")

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for e in range(epochs):
            logger.info("Epoch %d/%d", e + 1, epochs)

            # Resetting states to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=["loss(G)", "loss(D)"])

            for batch in batches:
                # Iterate through all possible critic steps
                for _ in range(critic_steps):
                    # Performs the optimization step over the discriminator
                    self.D_step(batch)

                # Performs the optimization step over the generator
                self.G_step(batch)

                # Adding corresponding values to the progress bar
                b.add(
                    1,
                    values=[
                        ("loss(G)", self.G_loss.result()),
                        ("loss(D)", self.D_loss.result()),
                    ],
                )

            # Dumps the losses to history
            self.history["G_loss"].append(self.G_loss.result().numpy())
            self.history["D_loss"].append(self.D_loss.result().numpy())

            logger.to_file(
                "Loss(G): %s | Loss(D): %s",
                self.G_loss.result().numpy(),
                self.D_loss.result().numpy(),
            )
