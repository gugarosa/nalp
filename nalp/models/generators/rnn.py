from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.core.model import Generator

logger = l.get_logger(__name__)


class RNNGenerator(Generator):
    """An RNNGenerator class is the one in charge of Recurrent Neural Networks vanilla implementation.

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

        # Overrides its parent class with any custom arguments if needed
        super(RNNGenerator, self).__init__(name='G_rnn')

        # Creates a property for holding the used encoder
        self.encoder = encoder

        # Creates an embedding layer
        self.embedding = layers.Embedding(
            vocab_size, embedding_size, name='embedding')

        # Creates a simple RNN cell
        self.cell = layers.SimpleRNNCell(hidden_size, name='rnn_cell')

        # Creates the RNN loop itself
        self.rnn = layers.RNN(self.cell, name='rnn_layer',
                              
                              stateful=True)

        # Creates the linear (Dense) layer
        self.linear = layers.Dense(vocab_size, name='out')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

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
