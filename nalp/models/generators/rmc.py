"""Relational Memory Core generator.
"""
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding

from nalp.core import Generator
from nalp.encoders.integer import IntegerEncoder
from nalp.models.layers.relational_memory_cell import RelationalMemoryCell
from nalp.utils import logging

logger = logging.get_logger(__name__)


class RMCGenerator(Generator):
    """An RMCGenerator class is the one in charge of
    Relational Recurrent Neural Networks vanilla implementation.

    References:
        A. Santoro, et al. Relational recurrent neural networks.
        Advances in neural information processing systems (2018).

    """

    def __init__(
        self,
        encoder: Optional[IntegerEncoder] = None,
        vocab_size: Optional[int] = 1,
        embedding_size: Optional[int] = 32,
        n_slots: Optional[int] = 3,
        n_heads: Optional[int] = 5,
        head_size: Optional[int] = 10,
        n_blocks: Optional[int] = 1,
        n_layers: Optional[int] = 3,
    ) -> None:
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

        """

        logger.info("Overriding class: Generator -> RMCGenerator.")

        super(RMCGenerator, self).__init__(name="G_rmc")

        # Creates a property for holding the used encoder
        self.encoder = encoder

        # Creates an embedding layer
        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        # Creates a relational memory cell
        self.cell = RelationalMemoryCell(
            n_slots, n_heads, head_size, n_blocks, n_layers, name="rmc_cell"
        )

        # Creates the RNN loop itself
        self.rnn = RNN(
            self.cell, name="rnn_layer", return_sequences=True, stateful=True
        )

        # Creates the linear (Dense) layer
        self.linear = Dense(vocab_size, name="out")

        logger.info("Class overrided.")

    @property
    def encoder(self) -> IntegerEncoder:
        """An encoder generic object."""

        return self._encoder

    @encoder.setter
    def encoder(self, encoder: IntegerEncoder) -> None:
        self._encoder = encoder

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        # Checks if the input tensor has a batch size
        if x.shape[0] is not None:
            # If yes, applies as a property
            self.batch_size = x.shape[0]

        # Firstly, we apply the embedding layer
        x = self.embedding(x)

        # We need to apply the input into the first recurrent layer
        x = self.rnn(
            x, initial_state=self.cell.get_initial_state(batch_size=self.batch_size)
        )

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        return x
