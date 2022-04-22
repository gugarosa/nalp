"""Linear generator.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense

from nalp.core import Generator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class LinearGenerator(Generator):
    """A LinearGenerator class stands for the
    linear generative part of a Generative Adversarial Network.

    """

    def __init__(
        self,
        input_shape: Optional[Tuple[int, ...]] = (784,),
        noise_dim: Optional[int] = 100,
        n_samplings: Optional[int] = 3,
        alpha: Optional[float] = 0.01,
    ) -> None:
        """Initialization method.

        Args:
            input_shape: An input shape for the tensor.
            noise_dim: Amount of noise dimensions.
            n_samplings: Number of upsamplings to perform.
            alpha: LeakyReLU activation threshold.

        """

        logger.info("Overriding class: Generator -> LinearGenerator.")

        super(LinearGenerator, self).__init__(name="G_linear")

        # Defining a property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a property for the input noise dimension
        self.noise_dim = noise_dim

        # Defining a list for holding the linear layers
        self.linear = [
            Dense(128 * (i + 1), name=f"linear_{i}") for i in range(n_samplings)
        ]

        # Defining the output layer with a `tanh` activation for restraining interval to [-1, 1]
        self.out = Dense(input_shape[0], activation="tanh", name="out")

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """LeakyReLU activation threshold."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def noise_dim(self) -> int:
        """Amount of noise dimensions."""

        return self._noise_dim

    @noise_dim.setter
    def noise_dim(self, noise_dim: int) -> None:
        self._noise_dim = noise_dim

    def call(self, x: tf.Tensor, training: Optional[bool] = True) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.
            training: Whether architecture is under training or not.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        # For every possible linear layer
        for layer in self.linear:
            # Applies the layer with a LeakyReLU activation
            x = tf.nn.leaky_relu(layer(x), self.alpha)

        # Passing down the output layer
        x = self.out(x)

        return x
