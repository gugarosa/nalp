from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.models.base import Model

logger = l.get_logger(__name__)


class BiLSTM(Model):
    """A BiLSTM class is the one in charge of a bi-directional Long Short-Term Memory implementation.

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

        logger.info('Overriding class: Model -> BiLSTM.')

        # Overrides its parent class with any custom arguments if needed
        super(BiLSTM, self).__init__(name='bi_lstm')

        # Creates an embedding layer
        self.embedding = layers.Embedding(
            vocab_size, embedding_size, name='embedding')

        # Creates a forward LSTM cell
        cell_f = layers.LSTMCell(hidden_size, name='lstm_cell_f')

        # And a backward LSTM cell
        cell_b = layers.LSTMCell(hidden_size, name='lstm_cell_b')

        # Creates the forward RNN layer
        forward = layers.RNN(cell_f, name='forward_rnn',
                             return_sequences=True, stateful=True)

        # Creates the backward RNN layer
        backward = layers.RNN(cell_b, name='backward_rnn',
                              return_sequences=True, stateful=True, go_backwards=True)

        # Creates the bi-directional Layer
        self.bidirect = layers.Bidirectional(
            forward, backward_layer=backward, name='bidirectional')

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

        # Then, we pass it to the bi-directional layer
        x = self.bidirect(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        return x
