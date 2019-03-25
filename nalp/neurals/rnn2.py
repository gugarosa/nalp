import nalp.utils.decorators as d
import nalp.utils.logging as l
import numpy as np
import tensorflow as tf
from nalp.core.neural import Neural
from nalp.neurals.layers.linear import Linear


logger = l.get_logger(__name__)


class RNN(tf.keras.Model):
    """A RNN class is the one in charge of Recurrent Neural Networks vanilla implementation.
    
    References:
        http://psych.colorado.edu/~kimlab/Elman1990.pdf

    """

    def __init__(self, max_length=1, vocab_size=1, hidden_size=2, learning_rate=0.001, shape=None):
        """Initialization method.

        Args:
            max_length (int): The maximum length of the encoding.
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The amount of hidden neurons.
            learning_rate (float): A big or small addition on the optimizer steps.
            shape (list): A list containing in its first position the shape of the inputs (x)
                and on the second position, the shape of the labels (y).

        """

        logger.info('Overriding class: Neural -> RNN.')

        # Overrides its parent class with any custom arguments if needed
        super(RNN, self).__init__()

        self.cell = tf.keras.layers.SimpleRNNCell(hidden_size)

        self.rnn = tf.keras.layers.RNN(self.cell)

        self.o = Linear(vocab_size)

        logger.info('Class overrided.')

    def call(self, x):
        x = self.rnn(x)
        x = self.o(x)

        return x


    