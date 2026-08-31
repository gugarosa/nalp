"""Generative Adversarial Network."""

from nalp.core import Adversarial
from nalp.models.discriminators import LinearDiscriminator
from nalp.models.generators import LinearGenerator


class GAN(Adversarial):
    """A GAN class is the one in charge of naïve Generative Adversarial Networks implementation.

    References:
        I. Goodfellow, et al. Generative adversarial nets.
        Advances in neural information processing systems (2014).

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
            input_shape: An input shape for the Generator.
            noise_dim: Amount of noise dimensions for the Generator.
            n_samplings: Number of down/up samplings to perform.
            alpha: LeakyReLU activation threshold.

        """

        D = LinearDiscriminator(n_samplings, alpha)
        G = LinearGenerator(input_shape, noise_dim, n_samplings, alpha)

        super().__init__(D, G, name="gan")
