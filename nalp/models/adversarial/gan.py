from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.wrappers.adversarial import AdversarialWrapper
from nalp.wrappers.standard import StandardWrapper

logger = l.get_logger(__name__)


class DiscriminatorGAN(StandardWrapper):
    """
    """

    def __init__(self, vocab_size=1, embedding_size=1):
        """Initialization method.

        """

        logger.info('Overriding class: StandardWrapper -> DiscriminatorGAN.')

        # Overrides its parent class with any custom arguments if needed
        super(DiscriminatorGAN, self).__init__(name='discriminator_gan')

        # Creates an embedding layer
        self.embedding = layers.Embedding(vocab_size, embedding_size, name='embedding')

        #
        self.conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='conv1')

        #
        self.leaky_relu = layers.LeakyReLU(name='leaky_relu')

        #
        self.dropout = layers.Dropout(0.3, name='drop')

        #
        self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='conv2')

        #
        self.linear = layers.Dense(1, name='dense')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        #
        x = self.embedding(x)
        
        #
        x = self.conv1(x)

        #
        x = self.leaky_relu(x)

        #
        x = self.dropout(x)

        #
        x = self.conv2(x)

        #
        x = self.leaky_relu(x)

        #
        x = self.dropout(x)

        #
        x = self.linear(x)

        return x

class GeneratorGAN(StandardWrapper):
    """
    """

    def __init__(self, vocab_size=1, embedding_size=1):
        """Initialization method.

        """

        logger.info('Overriding class: StandardWrapper -> GeneratorGAN.')

        # Overrides its parent class with any custom arguments if needed
        super(GeneratorGAN, self).__init__(name='generator_gan')

        # Creates an embedding layer
        self.embedding = layers.Embedding(vocab_size, embedding_size, name='embedding')

        #
        self.conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='conv1')

        #
        self.leaky_relu = layers.LeakyReLU(name='leaky_relu')

        #
        self.dropout = layers.Dropout(0.3, name='drop')

        #
        self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='conv2')

        #
        self.linear = layers.Dense(1, name='dense')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        #
        x = self.embedding(x)
        
        #
        x = self.conv1(x)

        #
        x = self.leaky_relu(x)

        #
        x = self.dropout(x)

        #
        x = self.conv2(x)

        #
        x = self.leaky_relu(x)

        #
        x = self.dropout(x)

        #
        x = self.linear(x)

        return x


class GAN(AdversarialWrapper):
    """A GAN class is the one in charge of Generative Adversarial Networks implementation.

    References:
        

    """

    def __init__(self, vocab_size=1, embedding_size=1):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: AdversarialWrapper -> GAN.')

        #
        D = DiscriminatorGAN(vocab_size=vocab_size, embedding_size=embedding_size)

        #
        G = GeneratorGAN(vocab_size=vocab_size, embedding_size=embedding_size)

        # Overrides its parent class with any custom arguments if needed
        super(GAN, self).__init__(D, G, name='gan')


    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """
