import math

import tensorflow as tf
from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.models.base import AdversarialModel, Model

logger = l.get_logger(__name__)


class Discriminator(Model):
    """A Discriminator class stands for the discriminative part of a Deep Convolutional Generative Adversarial Network.

    """

    def __init__(self, n_samplings, alpha, dropout_rate):
        """Initialization method.

        Args:
            n_samplings (int): Number of downsamplings to perform.
            alpha (float): LeakyReLU activation threshold.
            dropout_rate (float): Dropout activation rate.

        """

        logger.info('Overriding class: Model -> Discriminator.')

        # Overrides its parent class with any custom arguments if needed
        super(Discriminator, self).__init__(name='D_dcgan')

        # Defining a property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a list for holding the convolutional layers
        self.conv = []

        # Defining a list for holding the dropout layers
        self.drop = []

        # For every possible downsampling
        for i in range(n_samplings):
            # Appends a convolutional layer to the list
            self.conv.append(layers.Conv2D(64 * (i + 1), (5, 5), strides=(2, 2), padding='same'))

            # Appends a dropout layer to the list
            self.drop.append(layers.Dropout(dropout_rate))

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

        # For every possible convolutional and dropout layer
        for c, d in zip(self.conv, self.drop):
            # Applies the convolutional layer with a LeakyReLU activation and dropout
            x = d(tf.nn.leaky_relu(c(x), self.alpha), training=training)

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
        super(Generator, self).__init__(name='G_dcgan')

        # Defining an alpha property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a property for the input noise dimension
        self.noise_dim = noise_dim

        # Defining a property for the sampling factor used to calculate the upsampling
        self.sampling_factor = 2 ** (n_samplings - 1)

        # Defining a property for the initial size of the filter
        self.filter_size = int(input_shape[0] / self.sampling_factor)

        # Defining a list for holding the upsampling layers
        self.sampling = []

        # Defining a list for holding the batch normalization layers
        self.bn = []

        # For every possible upsampling
        for i in range(n_samplings, 0, -1):
            # If it is the first upsampling
            if i == n_samplings:
                # Appends a linear layer with a custom amount of units
                self.sampling.append(layers.Dense(self.filter_size ** 2 * 64 * self.sampling_factor, use_bias=False))

            # If it is the second upsampling
            elif i == n_samplings - 1:
                # Appends a convolutional layer with (1, 1) strides
                self.sampling.append(layers.Conv2DTranspose(64 * i, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            
            # If it is the rest of the upsamplings
            else:
                # Appends a convolutional layer with (2, 2) strides
                self.sampling.append(layers.Conv2DTranspose(64 * i, (5, 5), strides=(2, 2), padding='same', use_bias=False))

            # Appends a batch normalization layer to the list
            self.bn.append(layers.BatchNormalization())

        # Defining the output layer, which will be a convolutional transpose layer with `n_channels` filters
        self.out = layers.Conv2DTranspose(input_shape[3], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')


    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # For every possible layer in the list
        for i, (s, bn) in enumerate(zip(self.sampling, self.bn)):
            # Pass down the upsampling layer along with batch normalization and a LeakyReLU activation
            x = tf.nn.leaky_relu(bn(s(x), training=training), self.alpha)

            # If it is the first layer, e.g., linear
            if i == 0:
                # Reshapes the tensor for the convolutional layer
                x = tf.reshape(x, [x.shape[0], self.filter_size, self.filter_size, 64 * self.sampling_factor])

        # Passing down output layer
        x = self.out(x)

        return x


class DCGAN(AdversarialModel):
    """A DCGAN class is the one in charge of Deep Convolutional Generative Adversarial Networks implementation.

    References:
        A. Radford, L. Metz, S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. Preprint arXiv:1511.06434 (2015).

    """

    def __init__(self, input_shape=(28, 28, 1), noise_dim=100, n_samplings=3, alpha=0.3, dropout_rate=0.3):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the Generator.
            noise_dim (int): Amount of noise dimensions for the Generator.
            n_samplings (int): Number of down/up samplings to perform.
            alpha (float): LeakyReLU activation threshold.
            dropout_rate (float): Dropout activation rate.

        """

        logger.info('Overriding class: AdversarialModel -> DCGAN.')

        # Creating the discriminator network
        D = Discriminator(n_samplings, alpha, dropout_rate)

        # Creating the generator network
        G = Generator(input_shape, noise_dim, n_samplings, alpha)

        # Overrides its parent class with any custom arguments if needed
        super(DCGAN, self).__init__(D, G, name='dcgan')

        logger.info(
            f'Input: {input_shape} | Noise: {noise_dim} | Number of Samplings: {n_samplings} | Activation Rate: {alpha} | Dropout Rate: {dropout_rate}.')
