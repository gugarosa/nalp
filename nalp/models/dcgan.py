"""Deep Convolutional Generative Adversarial Network."""

from nalp.core import Adversarial
from nalp.models.discriminators import ConvDiscriminator
from nalp.models.generators import ConvGenerator


class DCGAN(Adversarial):
    """A DCGAN class is the one in charge of Deep Convolutional Generative Adversarial Networks implementation.

    References:
        A. Radford, L. Metz, S. Chintala.
        Unsupervised representation learning with deep convolutional generative adversarial networks.
        Preprint arXiv:1511.06434 (2015).

    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (28, 28, 1),
        noise_dim: int = 100,
        n_samplings: int = 3,
        alpha: float = 0.3,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialization method.

        Args:
            input_shape: An input shape for the Generator.
            noise_dim: Amount of noise dimensions for the Generator.
            n_samplings: Number of down/up samplings to perform.
            alpha: LeakyReLU activation threshold.
            dropout_rate: Dropout activation rate.

        """

        D = ConvDiscriminator(n_samplings, alpha, dropout_rate)
        G = ConvGenerator(input_shape, noise_dim, n_samplings, alpha)

        super().__init__(D, G, name="dcgan")
