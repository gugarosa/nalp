import tensorflow as tf
from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.models.base import AdversarialModel, Model

logger = l.get_logger(__name__)


class Discriminator(Model):
    """A Discriminator class stands for the discriminative part of a Generative Adversarial Network.

    """

    def __init__(self, alpha=0.01):
        """Initialization method.

        Args:
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: Model -> Discriminator.')

        # Overrides its parent class with any custom arguments if needed
        super(Discriminator, self).__init__(name='D_gan')

        # Defining an alpha property for the LeakyReLU activation
        self.alpha = alpha

        # Defining the first linear layer
        self.linear1 = layers.Dense(512)

        # Defining the second linear layer
        self.linear2 = layers.Dense(256)

        # Defining the third linear layer
        self.linear3 = layers.Dense(128)

        # Defining the output as a logit unit that decides whether input is real or fake
        self.out = layers.Dense(1)

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # Passing down first linear layer with LeakyReLU activation
        x = tf.nn.leaky_relu(self.linear1(x), self.alpha)

        # Passing down second linear layer with LeakyReLU activation
        x = tf.nn.leaky_relu(self.linear2(x), self.alpha)

        # Passing down third linear layer with LeakyReLU activation
        x = tf.nn.leaky_relu(self.linear3(x), self.alpha)

        # Passing down the output layer
        x = self.out(x)

        return x


class Generator(Model):
    """A Generator class stands for the generative part of a Generative Adversarial Network.

    """

    def __init__(self, n_input=100, n_output=784, alpha=0.01):
        """Initialization method.

        Args:
            n_input (int): Number of input (noise) dimension.
            n_output (int): Number of output units.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: Model -> Generator.')

        # Overrides its parent class with any custom arguments if needed
        super(Generator, self).__init__(name='G_gan')

        # Defining an alpha property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a property for the input noise dimension
        self.n_input = n_input

        # Defining the first linear layer
        self.linear1 = layers.Dense(128)

        # Defining the second linear layer
        self.linear2 = layers.Dense(256)

        # Defining the third linear layer
        self.linear3 = layers.Dense(512)

        # Defining the output layer with a `tanh` activation for restraining interval to [-1, 1]
        self.out = layers.Dense(n_output, activation='tanh')

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # Passing down first linear layer with LeakyReLU activation
        x = tf.nn.leaky_relu(self.linear1(x), self.alpha)

        # Passing down second linear layer with LeakyReLU activation
        x = tf.nn.leaky_relu(self.linear2(x), self.alpha)

        # Passing down third linear layer with LeakyReLU activation
        x = tf.nn.leaky_relu(self.linear3(x), self.alpha)

        # Passing down the output layer
        x = self.out(x)

        return x


class GAN(AdversarialModel):
    """A GAN class is the one in charge of naïve Generative Adversarial Networks implementation.

    References:
        I. Goodfellow, et al. Generative adversarial nets. Advances in neural information processing systems (2014).

    """

    def __init__(self, gen_input=100, gen_output=784, alpha=0.01):
        """Initialization method.

        Args:
            gen_input (int): Number of input (noise) dimension in the Generator.
            gen_output (int): Number of output units in the Generator.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: AdversarialModel -> GAN.')

        # Creating the discriminator network
        D = Discriminator(alpha=alpha)

        # Creating the generator network
        G = Generator(n_input=gen_input, n_output=gen_output, alpha=alpha)

        # Overrides its parent class with any custom arguments if needed
        super(GAN, self).__init__(D, G, name='gan')
