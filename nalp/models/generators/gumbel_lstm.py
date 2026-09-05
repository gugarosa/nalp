"""Gumbel Long Short-Term Memory generator."""

import tensorflow as tf

from nalp.encoders.integer import IntegerEncoder
from nalp.models.generators._gumbel import GumbelGeneratorMixin
from nalp.models.generators.lstm import LSTMGenerator
from nalp.models.layers import GumbelSoftmax


class GumbelLSTMGenerator(GumbelGeneratorMixin, LSTMGenerator):
    """A GumbelLSTMGenerator class is the one in charge of a
    generative Gumbel-based Long Short-Term Memory implementation.

    """

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        tau: float = 5.0,
    ) -> None:
        """Initialization method.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.
            tau: Gumbel-Softmax temperature parameter.

        """

        super().__init__(encoder, vocab_size, embedding_size, hidden_size)

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
