# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Deep Convolutional Generative Adversarial Network."""

from nalp.core.model import Adversarial
from nalp.models.discriminators.conv import ConvDiscriminator
from nalp.models.generators.conv import ConvGenerator


class DCGAN(Adversarial):
    """Train a deep convolutional generative adversarial network."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (28, 28, 1),
        noise_dim: int = 100,
        n_samplings: int = 3,
        alpha: float = 0.3,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize the convolutional discriminator and generator.

        Reference: A. Radford, L. Metz, S. Chintala.
        Unsupervised representation learning with deep convolutional generative adversarial networks.
        Preprint arXiv:1511.06434 (2015).

        Args:
            input_shape: Target image shape in height, width, and channels.
            noise_dim: Number of noise dimensions for the generator.
            n_samplings: Number of discriminator downsamplings and generator upsamplings.
            alpha: Negative slope of the LeakyReLU activation.
            dropout_rate: Dropout activation rate.

        """

        D = ConvDiscriminator(n_samplings, alpha, dropout_rate)
        G = ConvGenerator(input_shape, noise_dim, n_samplings, alpha)

        super().__init__(D, G, name="dcgan")
