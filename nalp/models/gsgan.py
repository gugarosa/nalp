import tensorflow as tf

import nalp.utils.logging as l
from nalp.core.model import Adversarial
from nalp.models.discriminators.lstm import LSTMDiscriminator
from nalp.models.generators.gumbel_lstm import GumbelLSTMGenerator

logger = l.get_logger(__name__)


class GSGAN(Adversarial):
    """A GSGAN class is the one in charge of Gumbel-Softmax Generative Adversarial Networks implementation.

    References:
        M. Kusner, J. Hernández-Lobato. Gans for sequences of discrete elements with the gumbel-softmax distribution.
        Preprint arXiv:1611.04051 (2016).

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

        logger.info('Overriding class: Adversarial -> GSGAN.')

        # Creating the discriminator network
        D = LSTMDiscriminator(vocab_size, embedding_size, hidden_size)

        # Creating the generator network
        G = GumbelLSTMGenerator(encoder, vocab_size, embedding_size, hidden_size, tau)

        # Overrides its parent class with any custom arguments if needed
        super(GSGAN, self).__init__(D, G, name='GSGAN')

        # Defining a property for holding the vocabulary size
        self.vocab_size = vocab_size

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    def compile(self, pre_optimizer, d_optimizer, g_optimizer):
        """Main building method.

        Args:
            pre_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the generator.
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.

        """

        # Creates an optimizer object for pre-training the generator
        self.P_optimizer = pre_optimizer

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

    def generate_batch(self, x):
        """Generates a batch of tokens by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            x (tf.Tensor): A tensor containing the inputs.

        Returns:
            A (batch_size, length) tensor of generated tokens and a
            (batch_size, length, vocab_size) tensor of predictions.

        """

        # Gathers the batch size and maximum sequence length
        batch_size, max_length = x.shape[0], x.shape[1]

        # Gathering the first token from the input tensor and expanding its last dimension
        start_batch = tf.expand_dims(x[:, 0], -1)

        # Creating an empty tensor for holding the Gumbel-Softmax predictions
        sampled_preds = tf.zeros([batch_size, 0, self.vocab_size])

        # Copying the sampled batch with the start batch tokens
        sampled_batch = start_batch

        # Resetting the network states
        self.G.reset_states()

        # For every possible generation
        for i in range(max_length):
            # Predicts the current token
            _, preds, start_batch = self.G(start_batch)

            # Concatenates the predictions with the tensor
            sampled_preds = tf.concat([sampled_preds, preds], 1)

            # Concatenates the sampled batch with the predicted batch
            sampled_batch = tf.concat([sampled_batch, start_batch], 1)

        # Ignoring the first column to get the target sampled batch
        sampled_batch = sampled_batch[:, 1:]

        return sampled_batch, sampled_preds

    def _discriminator_loss(self, y_real, y_fake):
        """Calculates the loss out of the discriminator architecture.

        Args:
            y_real (tf.Tensor): A tensor containing the real data targets.
            y_fake (tf.Tensor): A tensor containing the fake data targets.

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
            y_fake (tf.Tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the generator network.

        """

        # Calculating the generator loss
        loss = self.loss(tf.ones_like(y_fake), y_fake)

        return tf.reduce_mean(loss)

    @tf.function
    def G_pre_step(self, x, y):
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            x (tf.Tensor): A tensor containing the inputs.
            y (tf.Tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the logit-based predictions based on inputs
            logits, _, _ = self.G(x)

            # Calculate the loss
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits))

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.P_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    @tf.function
    def step(self, x, y):
        """Performs a single batch optimization step.

        Args:
            x (tf.Tensor): A tensor containing the inputs.
            y (tf.Tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # Generates new data, e.g., G(x)
            _, x_fake_probs = self.generate_batch(x)

            # Samples fake targets from the discriminator, e.g., D(G(x))
            y_fake = self.D(x_fake_probs)

            # Extends the target tensor to an one-hot encoding representation
            y = tf.one_hot(y, self.vocab_size)

            # Samples real targets from the discriminator, e.g., D(x)
            y_real = self.D(y)

            # Calculates the discriminator loss upon D(x) and D(G(x))
            D_loss = self._discriminator_loss(y_real, y_fake)

            # Calculates the generator loss upon D(G(x))
            G_loss = self._generator_loss(y_fake)

        # Calculate the gradients based on discriminator's loss for each training variable
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        # Calculate the gradients based on generator's loss for each training variable
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)

        # Applies the discriminator's gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(D_gradients, self.D.trainable_variables))

        # Applies the generator's gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(G_gradients, self.G.trainable_variables))

        # Updates the discriminator's loss state
        self.D_loss.update_state(D_loss)

        # Updates the generator's loss state
        self.G_loss.update_state(G_loss)

    def pre_fit(self, batches, epochs=100):
        """Pre-trains the model.

        Args:
            batches (Dataset): Pre-training batches containing samples.
            epochs (int): The maximum number of pre-training epochs.

        """

        logger.info('Pre-fitting generator ...')

        # Iterate through all generator epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting state to further append losses
            self.G_loss.reset_states()

            # Iterate through all possible pre-training batches
            for x_batch, y_batch in batches:
                # Performs the optimization step over the generator
                self.G_pre_step(x_batch, y_batch)

            logger.info(f'Loss(G): {self.G_loss.result().numpy()}')

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

            # Exponentially annealing the Gumbel-Softmax temperature
            self.G.tau = 5 ** ((epochs - e) / epochs)

            logger.info(
                f'Loss(G): {self.G_loss.result().numpy()} | Loss(D): {self.D_loss.result().numpy()}')
