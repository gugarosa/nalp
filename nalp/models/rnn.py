from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.neurals.simple import SimpleNeural

logger = l.get_logger(__name__)


class RNN(SimpleNeural):
    """An RNN class is the one in charge of Recurrent Neural Networks vanilla implementation.

    References:
        http://psych.colorado.edu/~kimlab/Elman1990.pdf

    """

    def __init__(self, vocab_size=1, embedding_size=1, hidden_size=1):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: Neural -> RNN.')

        # Overrides its parent class with any custom arguments if needed
        super(RNN, self).__init__(name='rnn')

        # Creates an embedding layer
        self.embedding = layers.Embedding(
            vocab_size, embedding_size, name='embedding')

        # Creates a simple RNN cell
        self.cell = layers.SimpleRNNCell(hidden_size, name='rnn_cell')

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
