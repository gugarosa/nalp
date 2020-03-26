import tensorflow as tf

import nalp.utils.logging as l
import nalp.utils.math as m
from nalp.core.model import Adversarial
from nalp.models.discriminators.lstm import LSTMDiscriminator
from nalp.models.generators.lstm import LSTMGenerator

logger = l.get_logger(__name__)


class GumbelLSTMGenerator(LSTMGenerator):
    """A GumbelLSTMGenerator class is the one in charge of a generative Gumbel-based Long Short-Term Memory implementation.

    References:
        E. Jang, S. Gu, B. Poole. Categorical reparameterization with gumbel-softmax. Preprint arXiv:1611.01144 (2016).

    """

    def __init__(self, encoder=None, vocab_size=1, embedding_size=32, hidden_size=64, tau=1):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder.
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.
            tau (float): Gumbel-Softmax temperature parameter.

        """

        logger.info('Overriding class: LSTMGenerator -> GumbelLSTMGenerator.')

        # Overrides its parent class with any custom arguments if needed
        super(GumbelLSTMGenerator, self).__init__(
            encoder, vocab_size, embedding_size, hidden_size)

        # Defining a property to hold the Gumbel-Softmax temperature parameter
        self.tau = tau

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # # Firstly, we apply the embedding layer
        # x = self.embedding(x)

        # # We need to apply the input into the first recurrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        # Adding a sampled Gumbel distribution to the output
        x += m.gumbel_distribution(x.shape)


        x = tf.nn.softmax(1 / (x * self.tau))

        return x


class GSGAN(Adversarial):
    """A GSGAN class is the one in charge of Gumbel-Softmax Generative Adversarial Networks implementation.

    References:
        M. Kusner, J. Hernández-Lobato. Gans for sequences of discrete elements with the gumbel-softmax distribution. Preprint arXiv:1611.04051 (2016).

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
        D = LSTMDiscriminator(encoder, vocab_size, embedding_size, hidden_size)

        # Creating the generator network
        G = GumbelLSTMGenerator(encoder, vocab_size,
                                embedding_size, hidden_size, tau=1)

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
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits

        # Defining a loss metric for the generator
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Defining a loss metric for the discriminator
        self.D_loss = tf.metrics.Mean(name='D_loss')

    def discriminator_loss(self, y, y_fake):
        """Calculates the loss out of the discriminator architecture.

        Args:
            y (tf.Tensor): A tensor containing the real data targets.
            y_fake (tf.Tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the discriminator network.

        """

        # Calculates the real data loss
        real_loss = self.loss(tf.ones_like(y), y)

        # Calculates the fake data loss
        fake_loss = self.loss(tf.zeros_like(y_fake), y_fake)

        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    def generator_loss(self, y_fake):
        """Calculates the loss out of the generator architecture.

        Args:
            y_fake (tf.Tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the generator network.

        """

        # Calculating the generator loss
        loss = self.loss(tf.ones_like(y_fake), y_fake)

        return tf.reduce_mean(loss)

    @tf.function
    def step(self, x, y):
        """Performs a single batch optimization step.

        Args:
            x (tf.Tensor): A tensor containing the inputs.

        """

        # Defines a random noise signal as the generator's input
        z = tf.random.normal([x.shape[0], x.shape[1], 256])

        # Using tensorflow's gradient
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # x_fake = tf.nn.softmax(tf.divide(1, x_fake), -1)

            # print(x_fake.shape)

            # Samples fake targets from the discriminator, e.g., D(G(z))
            y_fake = self.D(x_fake)

            x = tf.one_hot(x, 223)

            # print(x.shape)

            # Samples real targets from the discriminator, e.g., D(x)
            y = self.D(x)

            # Calculates the discriminator loss upon D(x) and D(G(z))
            D_loss = self.discriminator_loss(y, y_fake)

            # Calculates the generator loss upon D(G(z))
            G_loss = self.generator_loss(y_fake)

        # # Calculate the gradients based on generator's loss for each training variable
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)

        # # Calculate the gradients based on discriminator's loss for each training variable
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        # Applies the generator's gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(G_gradients, self.G.trainable_variables))

        # # Applies the discriminator's gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(D_gradients, self.D.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(G_loss)

        # # Updates the discriminator's loss state
        self.D_loss.update_state(D_loss)

    def fit(self, batches, epochs=100):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting states to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Iterate through all possible training batches
            for x_batch, y_batch in batches:
                # Performs the optimization step
                self.step(x_batch, y_batch)

            logger.info(
                f'Loss(G): {self.G_loss.result().numpy()} | Loss(D): {self.D_loss.result().numpy()}')
