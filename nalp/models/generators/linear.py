"""Linear generator.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense

from nalp.core import Generator


class LinearGenerator(Generator):
    """A LinearGenerator class stands for the
    linear generative part of a Generative Adversarial Network.

    """

    def __init__(
        self,
        input_shape: tuple[int, ...] = (784,),
        noise_dim: int = 100,
        n_samplings: int = 3,
        alpha: float = 0.01,
    ) -> None:
        """Initialization method.

        Args:
            input_shape: An input shape for the tensor.
            noise_dim: Amount of noise dimensions.
            n_samplings: Number of upsamplings to perform.
            alpha: LeakyReLU activation threshold.

        """

        super().__init__(name="G_linear")

        self.alpha = alpha
        self.noise_dim = noise_dim

        self.linear = [
            Dense(128 * (i + 1), name=f"linear_{i}") for i in range(n_samplings)
        ]

        self.out = Dense(input_shape[0], activation="tanh", name="out")

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.
            training: Whether architecture is under training or not.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        for layer in self.linear:
            x = tf.nn.leaky_relu(layer(x), self.alpha)

        x = self.out(x)

        return x
