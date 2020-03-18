import tensorflow as tf
from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.models.base import AdversarialModel, Model

logger = l.get_logger(__name__)


class Discriminator(Model):
    """A Discriminator class stands for the discriminative part of a Generative Adversarial Network.

    """

    def __init__(self, n_samplings, alpha):
        """Initialization method.

        Args:
            n_samplings (int): Number of downsamplings to perform.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: Model -> Discriminator.')

        # Overrides its parent class with any custom arguments if needed
        super(Discriminator, self).__init__(name='D_gan')

        # Defining a property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a list for holding the linear layers
        self.linear = []

        # For every possible downsampling
        for i in range(n_samplings, 0, -1):
            # Appends a linear layer to the list
            self.linear.append(layers.Dense(128 * i))

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

        # For every possible linear layer
        for l in self.linear:
            # Applies the layer with a LeakyReLU activation
            x = tf.nn.leaky_relu(l(x), self.alpha)

        # Passing down the output layer
        x = self.out(x)

        return x


class Generator(Model):
    """A Generator class stands for the generative part of a Generative Adversarial Network.

    """

    def __init__(self, input_shape, noise_dim, n_samplings, alpha):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the tensor.
            noise_dim (int): Amount of noise dimensions.
            n_samplings (int): Number of upsamplings to perform.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: Model -> Generator.')

        # Overrides its parent class with any custom arguments if needed
        super(Generator, self).__init__(name='G_gan')

        # Defining a property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a property for the input noise dimension
        self.noise_dim = noise_dim

        # Defining a list for holding the linear layers
        self.linear = []

        # For every possible upsampling
        for i in range(n_samplings):
            # Appends a linear layer to the list
            self.linear.append(layers.Dense(128 * (i + 1)))

        # Defining the output layer with a `tanh` activation for restraining interval to [-1, 1]
        self.out = layers.Dense(input_shape[0], activation='tanh')

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # For every possible linear layer
        for l in self.linear:
            # Applies the layer with a LeakyReLU activation
            x = tf.nn.leaky_relu(l(x), self.alpha)

        # Passing down the output layer
        x = self.out(x)

        return x


class GAN(AdversarialModel):
    """A GAN class is the one in charge of naïve Generative Adversarial Networks implementation.

    References:
        I. Goodfellow, et al. Generative adversarial nets. Advances in neural information processing systems (2014).

    """

    def __init__(self, input_shape=(784,), noise_dim=100, n_samplings=3, alpha=0.01):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the Generator.
            noise_dim (int): Amount of noise dimensions for the Generator.
            n_samplings (int): Number of down/up samplings to perform.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: AdversarialModel -> GAN.')

        # Creating the discriminator network
        D = Discriminator(n_samplings, alpha)

        # Creating the generator network
        G = Generator(input_shape, noise_dim, n_samplings, alpha)

        # Overrides its parent class with any custom arguments if needed
        super(GAN, self).__init__(D, G, name='gan')

        logger.info(
            f'Input: {input_shape} | Noise: {noise_dim} | Number of Samplings: {n_samplings} | Activation Rate: {alpha}.')
