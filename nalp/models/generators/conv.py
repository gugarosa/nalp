"""Convolutional generator.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Dense

from nalp.core import Generator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class ConvGenerator(Generator):
    """A ConvGenerator class stands for the
    convolutional generative part of a Generative Adversarial Network.

    """

    def __init__(
        self,
        input_shape: Optional[Tuple[int, int, int]] = (28, 28, 1),
        noise_dim: Optional[int] = 100,
        n_samplings: Optional[int] = 3,
        alpha: Optional[float] = 0.3,
    ) -> None:
        """Initialization method.

        Args:
            input_shape: An input shape for the tensor.
            noise_dim: Amount of noise dimensions.
            n_samplings: Number of upsamplings to perform.
            alpha: LeakyReLU activation threshold.

        """

        logger.info("Overriding class: Generator -> ConvGenerator.")

        super(ConvGenerator, self).__init__(name="G_conv")

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

    @property
    def sampling_factor(self) -> int:
        """Sampling factor used to calculate the upsampling."""

        return self._sampling_factor

    @sampling_factor.setter
    def sampling_factor(self, sampling_factor: int) -> None:
        self._sampling_factor = sampling_factor

    @property
    def filter_size(self) -> int:
        """Initial size of the filter."""

        return self._filter_size

    @filter_size.setter
    def filter_size(self, filter_size: int) -> None:
        self._filter_size = filter_size

    def call(self, x: tf.Tensor, training: Optional[bool] = True) -> tf.Tensor:
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
