from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.wrappers.standard import StandardWrapper

logger = l.get_logger(__name__)


class LSTM(StandardWrapper):
    """A LSTM class is the one in charge of Long Short-Term Memory implementation.

    References:
        S. Hochreiter, Jürgen Schmidhuber. Long short-term memory. Neural computation 9.8 (1997).

    """

    def __init__(self, vocab_size=1, embedding_size=1, hidden_size=1):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: StandardWrapper -> LSTM.')

        # Overrides its parent class with any custom arguments if needed
        super(LSTM, self).__init__(name='lstm')

        # Creates an embedding layer
        self.embedding = layers.Embedding(
            vocab_size, embedding_size, name='embedding')

        # Creates a LSTM cell
        self.cell = layers.LSTMCell(hidden_size, name='lstm_cell')

        # Creates the RNN loop itself
        self.rnn = layers.RNN(self.cell, name='rnn_layer',
                              return_sequences=True,
                              stateful=True)

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
