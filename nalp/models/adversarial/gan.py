import tensorflow as tf
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

        self.conv1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
        self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.conv2_bn = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), use_bias=False)
        self.conv3_bn = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(1, (3, 3))


    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """
        
        conv1 = tf.nn.leaky_relu(self.conv1(x))
        conv2 = self.conv2(conv1)
        conv2_bn = self.conv2_bn(conv2, training=training)
        conv3 = self.conv3(conv2_bn)
        conv3_bn = self.conv3_bn(conv3, training=training)
        conv4 = self.conv4(conv3_bn)
        discriminator_logits = tf.squeeze(conv4, axis=[1, 2])
        return discriminator_logits

class GeneratorGAN(StandardWrapper):
    """
    """

    def __init__(self, vocab_size=1, embedding_size=1):
        """Initialization method.

        """

        logger.info('Overriding class: StandardWrapper -> GeneratorGAN.')

        # Overrides its parent class with any custom arguments if needed
        super(GeneratorGAN, self).__init__(name='generator_gan')

        self.conv1 = layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
        self.conv1_bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
        self.conv2_bn = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.conv3_bn = layers.BatchNormalization()
        self.conv4 = layers.Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same')

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        
        conv1 = self.conv1(x)
        conv1_bn = self.conv1_bn(conv1, training=training)
        conv1 = tf.nn.relu(conv1_bn)
        
        conv2 = self.conv2(conv1)
        conv2_bn = self.conv2_bn(conv2, training=training)
        conv2 = tf.nn.relu(conv2_bn)
        
        conv3 = self.conv3(conv2)
        conv3_bn = self.conv3_bn(conv3, training=training)
        conv3 = tf.nn.relu(conv3_bn)
        
        conv4 = self.conv4(conv3)
        generated_data = tf.nn.sigmoid(conv4)
        return generated_data


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
