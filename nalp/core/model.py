"""Model-related classes.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class Discriminator(Model):
    """A Discriminator class is responsible for easily-implementing the discriminative part of
    a neural network, when custom training or additional sets are not needed.

    """

    def __init__(self, name=''):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Discriminator, self).__init__(name=name)

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError


class Generator(Model):
    """A Generator class is responsible for easily-implementing the generative part of
    a neural network, when custom training or additional sets are not needed.

    """

    def __init__(self, name=''):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Generator, self).__init__(name=name)

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def generate_text(self, start, length=100, temperature=1.0):
        """Generates text by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            start (str): The start string to generate the text.
            length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Returns:
            A list of generated text.

        """

        logger.debug('Generating text with length: %d ...', length)

        # Encoding the start string into tokens
        start_tokens = self.encoder.encode(start)

        # Expanding the first dimension of tensor
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(length):
            # Predicts the current token
            preds = self(start_tokens)

            # Removes the first dimension of the tensor
            preds = tf.squeeze(preds, 0)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a predicted token
            sampled_token = tf.random.categorical(preds, 1)[-1, 0].numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims([sampled_token], 0)

            # Appends the sampled token to the list
            sampled_tokens.append(sampled_token)

        # Decodes the list into raw text
        text = self.encoder.decode(sampled_tokens)

        return text


class Adversarial(Model):
    """An Adversarial class is responsible for customly
    implementing Generative Adversarial Networks.

    """

    def __init__(self, discriminator, generator, name=''):
        """Initialization method.

        Args:
            discriminator (Discriminator): Network's discriminator architecture.
            generator (Generator): Network's generator architecture.
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Adversarial, self).__init__(name=name)

        # Defining the discriminator network
        self.D = discriminator

        # Defining the generator network
        self.G = generator

    @property
    def D(self):
        """Discriminator: Discriminator architecture.

        """

        return self._D

    @D.setter
    def D(self, D):
        self._D = D

    @property
    def G(self):
        """Generator: Generator architecture.

        """

        return self._G

    @G.setter
    def G(self, G):
        self._G = G

    def compile(self, d_optimizer, g_optimizer):
        """Main building method.

        Args:
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.

        """

        # Creates an optimizer object for the discriminator
        self.D_optimizer = d_optimizer

        # Creates an optimizer object for the generator
        self.G_optimizer = g_optimizer

        # Defining the loss function
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits

        # Defining a loss metric for the discriminator
        self.D_loss = tf.metrics.Mean(name='D_loss')

        # Defining a loss metric for the generator
        self.G_loss = tf.metrics.Mean(name='G_loss')

    def _discriminator_loss(self, y_real, y_fake):
        """Calculates the loss out of the discriminator architecture.

        Args:
            y_real (tf.tensor): A tensor containing the real data targets.
            y_fake (tf.tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the discriminator network.

        """

        # Calculates the real data loss
        real_loss = self.loss(tf.ones_like(y_real), y_real)

        # Calculates the fake data loss
        fake_loss = self.loss(tf.zeros_like(y_fake), y_fake)

        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    def _generator_loss(self, y_fake):
        """Calculates the loss out of the generator architecture.

        Args:
            y_fake (tf.tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the generator network.

        """

        # Calculating the generator loss
        loss = self.loss(tf.ones_like(y_fake), y_fake)

        return tf.reduce_mean(loss)

    @tf.function
    def step(self, x):
        """Performs a single batch optimization step.

        Args:
            x (tf.tensor): A tensor containing the inputs.

        """

        # Defines a random noise signal as the generator's input
        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

        # Using tensorflow's gradient
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # Samples fake targets from the discriminator, e.g., D(G(z))
            y_fake = self.D(x_fake)

            # Samples real targets from the discriminator, e.g., D(x)
            y_real = self.D(x)

            # Calculates the generator loss upon D(G(z))
            G_loss = self._generator_loss(y_fake)

            # Calculates the discriminator loss upon D(x) and D(G(z))
            D_loss = self._discriminator_loss(y_real, y_fake)

        # Calculate the gradients based on generator's loss for each training variable
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)

        # Calculate the gradients based on discriminator's loss for each training variable
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        # Applies the generator's gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(G_gradients, self.G.trainable_variables))

        # Applies the discriminator's gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(D_gradients, self.D.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(G_loss)

        # Updates the discriminator's loss state
        self.D_loss.update_state(D_loss)

    def fit(self, batches, epochs=100):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Iterate through all epochs
        for e in range(epochs):
            logger.info('Epoch %d/%d', e+1, epochs)

            # Resetting states to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)', 'loss(D)'])

            # Iterate through all possible training batches
            for batch in batches:
                # Performs the optimization step
                self.step(batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result()),
                                 ('loss(D)', self.D_loss.result())])

            logger.file('Loss(G): %s | Loss(D): %s',
                        self.G_loss.result().numpy(), self.D_loss.result().numpy())
