"""Generative Adversarial Network.
"""

from typing import Optional, Tuple

from nalp.core import Adversarial
from nalp.models.discriminators import LinearDiscriminator
from nalp.models.generators import LinearGenerator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class GAN(Adversarial):
    """A GAN class is the one in charge of naïve Generative Adversarial Networks implementation.

    References:
        I. Goodfellow, et al. Generative adversarial nets.
        Advances in neural information processing systems (2014).

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
            input_shape: An input shape for the Generator.
            noise_dim: Amount of noise dimensions for the Generator.
            n_samplings: Number of down/up samplings to perform.
            alpha: LeakyReLU activation threshold.

        """

        logger.info("Overriding class: Adversarial -> GAN.")

        # Creating the discriminator network
        D = LinearDiscriminator(n_samplings, alpha)

        # Creating the generator network
        G = LinearGenerator(input_shape, noise_dim, n_samplings, alpha)

        super(GAN, self).__init__(D, G, name="gan")

        logger.debug(
            "Input: %s | Noise: %d | " "Number of samplings: %d | Activation rate: %s.",
            input_shape,
            noise_dim,
            n_samplings,
            alpha,
        )
        logger.info("Class overrided.")
