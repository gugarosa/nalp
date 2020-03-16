import tensorflow as tf
from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.wrappers.adversarial import AdversarialWrapper
from nalp.wrappers.standard import StandardWrapper

logger = l.get_logger(__name__)


class Discriminator(StandardWrapper):
    """
    """

    def __init__(self, vocab_size=1, embedding_size=1):
        """Initialization method.

        """

        logger.info('Overriding class: StandardWrapper -> Discriminator.')

        # Overrides its parent class with any custom arguments if needed
        super(Discriminator, self).__init__(name='D_gan')

        self.conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)



    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """
        
        conv1 = tf.nn.dropout(tf.nn.leaky_relu(self.conv1(x)), 0.3)
        conv2 = tf.nn.dropout(tf.nn.leaky_relu(self.conv2(conv1)), 0.3)
        discriminator_logits = self.dense(conv2)
        return discriminator_logits

class Generator(StandardWrapper):
    """
    """

    def __init__(self, vocab_size=1, embedding_size=1):
        """Initialization method.

        """

        logger.info('Overriding class: StandardWrapper -> Generator.')

        # Overrides its parent class with any custom arguments if needed
        super(Generator, self).__init__(name='G_gan')

        self.dense = layers.Dense(7*7*256, use_bias=False, input_shape=(100,))

        self.conv1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)

        self.conv2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)

        self.conv3 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

        self.bn_1 = layers.BatchNormalization()

        self.bn_2 = layers.BatchNormalization()

        self.bn_3 = layers.BatchNormalization()



    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """


        x = tf.nn.leaky_relu(self.bn_1(self.dense(x)))

        x = tf.reshape(x, [x.shape[0], 7, 7, 256])

        x = tf.nn.leaky_relu(self.bn_2(self.conv1(x)))

        x = tf.nn.leaky_relu(self.bn_3(self.conv2(x)))

        generated_data = self.conv3(x)

        return generated_data


class DCGAN(AdversarialWrapper):
    """A GAN class is the one in charge of Generative Adversarial Networks implementation.

    References:
        A. Radford, L. Metz, S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. Preprint arXiv:1511.06434 (2015).

    """

    def __init__(self, vocab_size=1, embedding_size=1):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: AdversarialWrapper -> GAN.')

        # Creating the discriminator network
        D = Discriminator(vocab_size=vocab_size, embedding_size=embedding_size)

        # Creating the generator network
        G = Generator(vocab_size=vocab_size, embedding_size=embedding_size)

        # Overrides its parent class with any custom arguments if needed
        super(DCGAN, self).__init__(D, G, name='dcgan')
