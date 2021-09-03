"""Generative Adversarial Network.
"""

import nalp.utils.logging as l
from nalp.core import Adversarial
from nalp.models.discriminators import LinearDiscriminator
from nalp.models.generators import LinearGenerator

logger = l.get_logger(__name__)


class GAN(Adversarial):
    """A GAN class is the one in charge of naïve Generative Adversarial Networks implementation.

    References:
        I. Goodfellow, et al. Generative adversarial nets.
        Advances in neural information processing systems (2014).

    """

    def __init__(self, input_shape=(784,), noise_dim=100, n_samplings=3, alpha=0.01):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the Generator.
            noise_dim (int): Amount of noise dimensions for the Generator.
            n_samplings (int): Number of down/up samplings to perform.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: Adversarial -> GAN.')

        # Creating the discriminator network
        D = LinearDiscriminator(n_samplings, alpha)

        # Creating the generator network
        G = LinearGenerator(input_shape, noise_dim, n_samplings, alpha)

        super(GAN, self).__init__(D, G, name='gan')

        logger.debug('Input: %s | Noise: %d | '
                     'Number of samplings: %d | Activation rate: %s.',
                     input_shape, noise_dim,
                     n_samplings, alpha)
        logger.info('Class overrided.')
