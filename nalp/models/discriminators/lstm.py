"""Long Short-Term Memory discriminator.
"""

from tensorflow.keras.layers import RNN, Dense, LSTMCell

import nalp.utils.logging as l
from nalp.core import Discriminator

logger = l.get_logger(__name__)


class LSTMDiscriminator(Discriminator):
    """A LSTMDiscriminator class is the one in charge of a
    discriminative Long Short-Term Memory implementation.

    References:
        S. Hochreiter, Jürgen Schmidhuber. Long short-term memory. Neural computation 9.8 (1997).

    """

    def __init__(self, embedding_size=32, hidden_size=64):
        """Initialization method.

        Args:
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: Discriminator -> LSTMDiscriminator.')

        super(LSTMDiscriminator, self).__init__(name='D_lstm')

        # Creates an embedding layer
        self.embedding = Dense(embedding_size, name='embedding')

        # Creates a LSTM cell
        self.cell = LSTMCell(hidden_size, name='lstm_cell')

        # Creates the RNN loop itself
        self.rnn = RNN(self.cell, name='rnn_layer',
                       return_sequences=True,
                       stateful=True)

        # And finally, defining the output layer
        self.out = Dense(1, name='out')

        logger.info('Class overrided.')

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
        x = self.out(x)

        return x
