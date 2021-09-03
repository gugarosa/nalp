"""Convolutional generator.
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Dense

import nalp.utils.logging as l
from nalp.core import Generator

logger = l.get_logger(__name__)


class ConvGenerator(Generator):
    """A ConvGenerator class stands for the
    convolutional generative part of a Generative Adversarial Network.

    """

    def __init__(self, input_shape=(28, 28, 1), noise_dim=100, n_samplings=3, alpha=0.3):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the tensor.
            noise_dim (int): Amount of noise dimensions.
            n_samplings (int): Number of upsamplings to perform.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: Generator -> ConvGenerator.')

        super(ConvGenerator, self).__init__(name='G_conv')

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

        # For every possible upsampling
        for i in range(n_samplings, 0, -1):
            # If it is the first upsampling
            if i == n_samplings:
                # Appends a linear layer with a custom amount of units
                self.sampling.append(Dense(self.filter_size ** 2 * 64 * self.sampling_factor,
                                           use_bias=False, name=f'linear_{i}'))

            # If it is the second upsampling
            elif i == n_samplings - 1:
                # Appends a convolutional layer with (1, 1) strides
                self.sampling.append(Conv2DTranspose(64 * i, (5, 5), strides=(1, 1),
                                                     padding='same', use_bias=False, name=f'conv_{i}'))

            # If it is the rest of the upsamplings
            else:
                # Appends a convolutional layer with (2, 2) strides
                self.sampling.append(Conv2DTranspose(64 * i, (5, 5), strides=(2, 2),
                                                     padding='same', use_bias=False, name=f'conv_{i}'))

        # Defining a list for holding the batch normalization layers
        self.bn = [BatchNormalization(name=f'bn_{i}')
                   for i in range(n_samplings, 0, -1)]

        # Defining the output layer, which will be a convolutional transpose layer with `n_channels` filters
        self.out = Conv2DTranspose(input_shape[2], (5, 5), strides=(2, 2),
                                   padding='same', use_bias=False,
                                   activation='tanh', name='out')

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: LeakyReLU activation threshold.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def noise_dim(self):
        """int: Amount of noise dimensions.

        """

        return self._noise_dim

    @noise_dim.setter
    def noise_dim(self, noise_dim):
        self._noise_dim = noise_dim

    @property
    def sampling_factor(self):
        """int: Sampling factor used to calculate the upsampling.

        """

        return self._sampling_factor

    @sampling_factor.setter
    def sampling_factor(self, sampling_factor):
        self._sampling_factor = sampling_factor

    @property
    def filter_size(self):
        """int: Initial size of the filter.

        """

        return self._filter_size

    @filter_size.setter
    def filter_size(self, filter_size):
        self._filter_size = filter_size

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.tensor): A tensorflow's tensor holding input data.
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
                x = tf.reshape(
                    x, [x.shape[0], self.filter_size, self.filter_size, 64 * self.sampling_factor])

        # Passing down output layer
        x = self.out(x)

        return x
