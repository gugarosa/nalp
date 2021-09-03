"""Recurrent Neural Network generator.
"""

from tensorflow.keras.layers import RNN, Dense, Embedding, SimpleRNNCell

import nalp.utils.logging as l
from nalp.core import Generator

logger = l.get_logger(__name__)


class RNNGenerator(Generator):
    """An RNNGenerator class is the one in charge of
    Recurrent Neural Networks vanilla implementation.

    References:
        J. Elman. Finding structure in time. Cognitive science 14.2 (1990).

    """

    def __init__(self, encoder=None, vocab_size=1, embedding_size=32, hidden_size=64):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder.
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: Generator -> RNNGenerator.')

        super(RNNGenerator, self).__init__(name='G_rnn')

        # Creates a property for holding the used encoder
        self.encoder = encoder

        # Creates an embedding layer
        self.embedding = Embedding(vocab_size, embedding_size, name='embedding')

        # Creates a simple RNN cell
        self.cell = SimpleRNNCell(hidden_size, name='rnn_cell')

        # Creates the RNN loop itself
        self.rnn = RNN(self.cell, name='rnn_layer',
                       return_sequences=True,
                       stateful=True)

        # Creates the linear (Dense) layer
        self.linear = Dense(vocab_size, name='out')

        logger.info('Class overrided.')

    @property
    def encoder(self):
        """obj: An encoder generic object.

        """

        return self._encoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # Firstly, we apply the embedding layer
        x = self.embedding(x)

        # We need to apply the input into the first recurrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        return x
