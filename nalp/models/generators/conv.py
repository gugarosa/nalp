"""Convolutional generator.
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Dense

from nalp.core import Generator


class ConvGenerator(Generator):
    """A ConvGenerator class stands for the
    convolutional generative part of a Generative Adversarial Network.

    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (28, 28, 1),
        noise_dim: int = 100,
        n_samplings: int = 3,
        alpha: float = 0.3,
    ) -> None:
        """Initialization method.

        Args:
            input_shape: An input shape for the tensor.
            noise_dim: Amount of noise dimensions.
            n_samplings: Number of upsamplings to perform.
            alpha: LeakyReLU activation threshold.

        """

        super().__init__(name="G_conv")

        self.alpha = alpha
        self.noise_dim = noise_dim

        self.sampling_factor = 2 ** (n_samplings - 1)
        self.filter_size = int(input_shape[0] / self.sampling_factor)

        self.sampling = []
        for i in range(n_samplings, 0, -1):
            if i == n_samplings:
                self.sampling.append(
                    Dense(
                        self.filter_size**2 * 64 * self.sampling_factor,
                        use_bias=False,
                        name=f"linear_{i}",
                    )
                )
            elif i == n_samplings - 1:
                self.sampling.append(
                    Conv2DTranspose(
                        64 * i,
                        (5, 5),
                        strides=(1, 1),
                        padding="same",
                        use_bias=False,
                        name=f"conv_{i}",
                    )
                )
            else:
                self.sampling.append(
                    Conv2DTranspose(
                        64 * i,
                        (5, 5),
                        strides=(2, 2),
                        padding="same",
                        use_bias=False,
                        name=f"conv_{i}",
                    )
                )

        self.bn = [
            BatchNormalization(name=f"bn_{i}") for i in range(n_samplings, 0, -1)
        ]

        self.out = Conv2DTranspose(
            input_shape[2],
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
            name="out",
        )

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.
            training: Whether architecture is under training or not.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        for i, (s, bn) in enumerate(zip(self.sampling, self.bn)):
            x = tf.nn.leaky_relu(bn(s(x), training=training), self.alpha)

            if i == 0:
                x = tf.reshape(
                    x,
                    [
                        x.shape[0],
                        self.filter_size,
                        self.filter_size,
                        64 * self.sampling_factor,
                    ],
                )

        x = self.out(x)

        return x
