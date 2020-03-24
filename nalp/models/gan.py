import nalp.utils.logging as l
from nalp.core.model import AdversarialModel
from nalp.models.discriminators.linear import LinearDiscriminator
from nalp.models.generators.linear import LinearGenerator

logger = l.get_logger(__name__)


class GAN(AdversarialModel):
    """A GAN class is the one in charge of naïve Generative Adversarial Networks implementation.

    References:
        I. Goodfellow, et al. Generative adversarial nets. Advances in neural information processing systems (2014).

    """

    def __init__(self, input_shape=(784,), noise_dim=100, n_samplings=3, alpha=0.01):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the Generator.
            noise_dim (int): Amount of noise dimensions for the Generator.
            n_samplings (int): Number of down/up samplings to perform.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: AdversarialModel -> GAN.')

        # Creating the discriminator network
        D = LinearDiscriminator(n_samplings, alpha)

        # Creating the generator network
        G = LinearGenerator(input_shape, noise_dim, n_samplings, alpha)

        # Overrides its parent class with any custom arguments if needed
        super(GAN, self).__init__(D, G, name='gan')

        logger.info(
            f'Input: {input_shape} | Noise: {noise_dim} | Number of Samplings: {n_samplings} | Activation Rate: {alpha}.')
