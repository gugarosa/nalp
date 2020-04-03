import nalp.utils.logging as l
from nalp.core.model import Adversarial
from nalp.models.discriminators.conv import ConvDiscriminator
from nalp.models.generators.conv import ConvGenerator

logger = l.get_logger(__name__)

class DCGAN(Adversarial):
    """A DCGAN class is the one in charge of Deep Convolutional Generative Adversarial Networks implementation.

    References:
        A. Radford, L. Metz, S. Chintala.
        Unsupervised representation learning with deep convolutional generative adversarial networks.
        Preprint arXiv:1511.06434 (2015).

    """

    def __init__(self, input_shape=(28, 28, 1), noise_dim=100, n_samplings=3, alpha=0.3, dropout_rate=0.3):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the Generator.
            noise_dim (int): Amount of noise dimensions for the Generator.
            n_samplings (int): Number of down/up samplings to perform.
            alpha (float): LeakyReLU activation threshold.
            dropout_rate (float): Dropout activation rate.

        """

        logger.info('Overriding class: Adversarial -> DCGAN.')

        # Creating the discriminator network
        D = ConvDiscriminator(n_samplings, alpha, dropout_rate)

        # Creating the generator network
        G = ConvGenerator(input_shape, noise_dim, n_samplings, alpha)

        # Overrides its parent class with any custom arguments if needed
        super(DCGAN, self).__init__(D, G, name='dcgan')

        logger.info(f'Input: {input_shape} | Noise: {noise_dim} | Number of Samplings: {n_samplings} | '
                    f'Activation Rate: {alpha} | Dropout Rate: {dropout_rate}.')
