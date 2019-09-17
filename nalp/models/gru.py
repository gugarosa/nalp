from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.neurals.simple import SimpleNeural

logger = l.get_logger(__name__)


class GRU(SimpleNeural):
    """A GRU class is the one in charge of Gated Recurrent Unit implementation.

    References:
        https://www.aclweb.org/anthology/D14-1179

    """

    def __init__(self, vocab_size=1, embedding_size=1, hidden_size=1):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: Neural -> GRU.')

        # Overrides its parent class with any custom arguments if needed
        super(GRU, self).__init__(name='gru')

        # Creates an embedding layer
        self.embedding = layers.Embedding(
            vocab_size, embedding_size, name='embedding')

        # Creates a GRU cell
        self.cell = layers.GRUCell(hidden_size, name='gru')

        # Creates the RNN loop itself
        self.rnn = layers.RNN(self.cell, name='rnn_layer',
                              return_sequences=True)

        # Creates the linear (Dense) layer
        self.linear = layers.Dense(vocab_size, name='dense')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # Firstly, we apply the embedding layer
        x = self.embedding(x)

        # We need to apply the input into the first recorrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        return x
