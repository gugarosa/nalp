import tensorflow as tf

import nalp.utils.logging as l
from nalp.core.model import Adversarial

logger = l.get_logger(__name__)


class RelGAN(Adversarial):
    """A RelGAN class is the one in charge of Relational Generative Adversarial Networks implementation.

    References:
        W. Nie, N. Narodytska, A. Patel. Relgan: Relational generative adversarial networks for text generation.
        International Conference on Learning Representations (2018).

    """

    def __init__(self, encoder=None, vocab_size=1, max_length=1, embedding_size=32, hidden_size=64, tau=5):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder for the generator.
            vocab_size (int): The size of the vocabulary for both discriminator and generator.
            max_length (int): Maximum length of the sequences for the discriminator.
            embedding_size (int): The size of the embedding layer for both discriminator and generator.
            hidden_size (int): The amount of hidden neurons for the generator.
            tau (float): Gumbel-Softmax temperature parameter.

        """

        logger.info('Overriding class: Adversarial -> RelGAN.')

        # Creating the discriminator network
        # D = 

        # Creating the generator network
        # G = 

        # Overrides its parent class with any custom arguments if needed
        super(RelGAN, self).__init__(D, G, name='RelGAN')

        # Defining a property for holding the vocabulary size
        self.vocab_size = vocab_size

    def compile(self, pre_optimizer, g_optimizer, d_optimizer):
        """Main building method.

        Args:
            pre_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the generator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.

        """

        # Creates an optimizer object for pre-training the generator
        self.P_optimizer = pre_optimizer

        # Creates an optimizer object for the generator
        self.G_optimizer = g_optimizer

        # Creates an optimizer object for the discriminator
        self.D_optimizer = d_optimizer

        # Defining the loss function
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits

        # Defining a loss metric for the generator
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Defining a loss metric for the discriminator
        self.D_loss = tf.metrics.Mean(name='D_loss')
