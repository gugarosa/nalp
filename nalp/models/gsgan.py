import tensorflow as tf

import nalp.utils.math as m
import nalp.utils.logging as l
from nalp.core.model import Adversarial
from nalp.models.discriminators.text import TextDiscriminator
from nalp.models.generators.lstm import LSTMGenerator

logger = l.get_logger(__name__)


class GumbelLSTMGenerator(LSTMGenerator):
    """
    """

    def __init__(self, encoder, vocab_size, embedding_size, hidden_size, tau=1):
        """
        """

        #
        super(GumbelLSTMGenerator, self).__init__(encoder, vocab_size, embedding_size, hidden_size)

        #
        self.tau = tau

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

        #
        x += m.gumbel_distribution(x.shape)

        #
        x /= self.tau

        return x




class GSGAN(Adversarial):
    """A GSGAN class is the one in charge of Gumbel-Softmax Generative Adversarial Networks implementation.

    References:
        

    """

    def __init__(self, encoder=None, vocab_size=1, max_length=1, embedding_size=32, hidden_size=64, n_filters=[64], filters_size=[1], dropout_rate=0.25, temperature=1):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder for the generator.
            vocab_size (int): The size of the vocabulary for both discriminator and generator.
            max_length (int): Maximum length of the sequences for the discriminator.
            embedding_size (int): The size of the embedding layer for both discriminator and generator.
            hidden_size (int): The amount of hidden neurons for the generator.
            n_filters (list): Number of filters to be applied in the discriminator.
            filters_size (list): Size of filters to be applied in the discriminator.
            dropout_rate (float): Dropout activation rate.
            temperature (float): Temperature value to sample the token.

        """

        logger.info('Overriding class: Adversarial -> GSGAN.')

        # Creating the discriminator network
        D = TextDiscriminator(
            vocab_size, max_length, embedding_size, n_filters, filters_size, dropout_rate)

        # Creating the generator network
        G = GumbelLSTMGenerator(encoder, vocab_size, embedding_size, hidden_size, tau=1)

        # Overrides its parent class with any custom arguments if needed
        super(GSGAN, self).__init__(D, G, name='GSGAN')

        # Defining a property for holding the vocabulary size
        self.vocab_size = vocab_size

        # Defining a property for holding the temperature
        self.T = temperature

    def compile(self, g_optimizer, d_optimizer):
        """Main building method.

        Args:
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.

        """

        # Creates an optimizer object for the generator
        self.G_optimizer = g_optimizer

        # Creates an optimizer object for the discriminator
        self.D_optimizer = d_optimizer

        # Defining the loss function
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits

        # Defining a loss metric for the generator
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Defining a loss metric for the discriminator
        self.D_loss = tf.metrics.Mean(name='D_loss')