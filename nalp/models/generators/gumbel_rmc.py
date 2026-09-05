"""Gumbel Relational Memory Core generator."""

import tensorflow as tf

from nalp.encoders.integer import IntegerEncoder
from nalp.models.generators._gumbel import GumbelGeneratorMixin
from nalp.models.generators.rmc import RMCGenerator
from nalp.models.layers import GumbelSoftmax


class GumbelRMCGenerator(GumbelGeneratorMixin, RMCGenerator):
    """A GumbelRMCGenerator class is the one in charge of a
    generative Gumbel-based Relational Memory Core implementation.

    """

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        n_slots: int = 3,
        n_heads: int = 5,
        head_size: int = 10,
        n_blocks: int = 1,
        n_layers: int = 3,
        tau: float = 5,
    ):
        """Initialization method.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            n_slots: Number of memory slots.
            n_heads: Number of attention heads.
            head_size: Size of each attention head.
            n_blocks: Number of feed-forward networks.
            n_layers: Amout of layers per feed-forward network.
            tau: Gumbel-Softmax temperature parameter.

        """

        super().__init__(
            encoder,
            vocab_size,
            embedding_size,
            n_slots,
            n_heads,
            head_size,
            n_blocks,
            n_layers,
        )

        self.tau = tau

        self.gumbel = GumbelSoftmax(name="gumbel")

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Logit-based predictions, Gumbel-Softmax outputs and predicted token.

        """

        x = super().call(x)

        x_g, y_g = self.gumbel(x, tau=self._tau_tensor)

        return x, x_g, y_g
