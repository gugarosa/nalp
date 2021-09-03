"""Model-related classes.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar

import nalp.utils.constants as c
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

    def generate_greedy_search(self, start, max_length=100):
        """Generates text by using greedy search, where the sampled
        token is always sampled according to the maximum probability.

        Args:
            start (str): The start string to generate the text.
            max_length (int): Maximum length of generated text.

        Returns:
            A list holding the generated text.

        """

        # Encoding the start string into tokens, while expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers the last timestep
            preds = self(start_tokens)
            preds = preds[:, -1, :]

            # Samples a predicted token
            sampled_token = tf.argmax(preds, 1).numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_temperature_sampling(self, start, max_length=100, temperature=1.0):
        """Generates text by using temperature sampling, where the sampled
        token is sampled according to a multinomial/categorical distribution.

        Args:
            start (str): The start string to generate the text.
            max_length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Returns:
            A list holding the generated text.

        """

        # Encoding the start string into tokens, while expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers last timestep
            preds = self(start_tokens)
            preds = preds[:, -1, :]

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a predicted token
            sampled_token = tf.random.categorical(preds, 1)[0].numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_top_sampling(self, start, max_length=100, k=0, p=0.0):
        """Generates text by using top-k and top-p sampling, where the sampled
        token is sampled according to the `k` most likely words distribution, as well
        as to the maximim cumulative probability `p`.

        Args:
            start (str): The start string to generate the text.
            max_length (int): Length of generated text.
            k (int): Indicates the amount of likely words.
            p (float): Maximum cumulative probability to be thresholded.

        Returns:
            A list holding the generated text.

        """

        # Encoding the start string into tokens, while expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers its last timestep
            preds = self(start_tokens)
            preds = preds[:, -1, :]

            # Checks if there is a provided `k`
            if k > 0:
                # Samples the top-k predictions and its indexes
                preds, preds_indexes = tf.math.top_k(preds, k)

            # If there is no provided `k`,
            # it means that we need to sort the predictions tensor
            else:
                # Gathers sorted predictions and its indexes
                preds, preds_indexes = tf.math.top_k(preds, preds.shape[-1])

            # Checks if there is a provided probability
            if p > 0.0:
                # Calculates the cumulative probability over the predictions' softmax
                cum_probs = tf.math.cumsum(tf.nn.softmax(preds), axis=-1)

                # Gathers a binary mask indicating whether indexes are below threshold
                ignored_indexes = cum_probs <= p

                # Also ensures that first index will always be true to prevent zero
                # tokens from being sampled
                ignored_indexes = tf.tensor_scatter_nd_update(ignored_indexes, [[0, 0]], [True])

                # Filters the predictions and its indexes
                preds = tf.expand_dims(preds[ignored_indexes], 0)
                preds_indexes = tf.expand_dims(preds_indexes[ignored_indexes], 0)

            # Samples an index from top-k logits and gathers the real token index
            index = tf.random.categorical(preds, 1)[0, 0]
            sampled_token = [preds_indexes[-1][index].numpy()]

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens


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

        super(Adversarial, self).__init__(name=name)

        # Defining the discriminator network
        self.D = discriminator

        # Defining the generator network
        self.G = generator

        # Defining the history
        self.history = {}

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

    @property
    def history(self):
        """dict: History dictionary.

        """

        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    def compile(self, d_optimizer, g_optimizer):
        """Main building method.

        Args:
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.

        """

        # Creates both optimizers
        self.D_optimizer = d_optimizer
        self.G_optimizer = g_optimizer

        # Defining the loss function
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits

        # Defining both loss metrics
        self.D_loss = tf.metrics.Mean(name='D_loss')
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Storing losses as history keys
        self.history['D_loss'] = []
        self.history['G_loss'] = []

    def _discriminator_loss(self, y_real, y_fake):
        """Calculates the loss out of the discriminator architecture.

        Args:
            y_real (tf.tensor): A tensor containing the real data targets.
            y_fake (tf.tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the discriminator network.

        """

        real_loss = self.loss(tf.ones_like(y_real), y_real)
        fake_loss = self.loss(tf.zeros_like(y_fake), y_fake)

        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    def _generator_loss(self, y_fake):
        """Calculates the loss out of the generator architecture.

        Args:
            y_fake (tf.tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the generator network.

        """

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

            # Samples fake targets D(G(z)) and real targets D(x) from the discriminator
            y_fake = self.D(x_fake)
            y_real = self.D(x)

            # Calculates both generator and discriminator losses
            G_loss = self._generator_loss(y_fake)
            D_loss = self._discriminator_loss(y_real, y_fake)

        # Calculate both gradients
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        # Applies both gradients using an optimizer
        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))

        # Updates the generator's and discriminator's loss states
        self.G_loss.update_state(G_loss)
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

        for e in range(epochs):
            logger.info('Epoch %d/%d', e+1, epochs)

            # Resetting states to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)', 'loss(D)'])

            for batch in batches:
                # Performs the optimization step
                self.step(batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result()),
                                 ('loss(D)', self.D_loss.result())])

            # Dumps the losses to history
            self.history['G_loss'].append(self.G_loss.result().numpy())
            self.history['D_loss'].append(self.D_loss.result().numpy())

            logger.to_file('Loss(G): %s | Loss(D): %s',
                        self.G_loss.result().numpy(), self.D_loss.result().numpy())
