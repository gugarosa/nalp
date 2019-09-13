import nalp.utils.logging as l
from nalp.core.neural import Neural
import tensorflow as tf

logger = l.get_logger(__name__)


class RNN(tf.keras.Model):
    """A RNN class is the one in charge of Recurrent Neural Networks vanilla implementation.
    
    References:
        http://psych.colorado.edu/~kimlab/Elman1990.pdf

    """

    def __init__(self, vocab_size=1, hidden_size=2):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The amount of hidden neurons.
            learning_rate (float): A big or small addition on the optimizer steps.

        """

        logger.info('Overriding class: Neural -> RNN.')

        # Overrides its parent class with any custom arguments if needed
        super(RNN, self).__init__(name='rnn')

        # Creates a simple RNN cell
        self.cell = tf.keras.layers.SimpleRNNCell(hidden_size, name='rnn_cell')

        # Creates the RNN loop itself
        self.rnn = tf.keras.layers.RNN(self.cell, name='rnn_layer')

        # Creates the linear (Dense) layer
        self.linear = tf.keras.layers.Dense(vocab_size, name='dense')

        # And finally, a softmax activation for life's easing
        self.softmax = tf.keras.layers.Softmax(name='softmax')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # We need to apply the input into the first recorrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        # Finally, we output its probabilites
        x = self.softmax(x)

        return x
