# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Generative Adversarial Network."""

from nalp.core.model import Adversarial
from nalp.models.discriminators.linear import LinearDiscriminator
from nalp.models.generators.linear import LinearGenerator


class GAN(Adversarial):
    """Train a generative adversarial network with linear components."""

    def __init__(
        self,
        input_shape: tuple[int, ...] = (784,),
        noise_dim: int = 100,
        n_samplings: int = 3,
        alpha: float = 0.01,
    ) -> None:
        """Initialize the linear discriminator and generator.

        Reference: I. Goodfellow, et al. Generative adversarial nets.
        Advances in neural information processing systems (2014).

        Args:
            input_shape: Output shape whose first dimension sets the generator's final feature count.
            noise_dim: Number of noise dimensions for the generator.
            n_samplings: Number of discriminator and generator hidden layers.
            alpha: Negative slope of the LeakyReLU activation.

        """

        D = LinearDiscriminator(n_samplings, alpha)
        G = LinearGenerator(input_shape, noise_dim, n_samplings, alpha)

        super().__init__(D, G, name="gan")
