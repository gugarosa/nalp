"""Sequence Generative Adversarial Network.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

import nalp.utils.constants as c
import nalp.utils.logging as l
from nalp.core import Adversarial
from nalp.models.discriminators import EmbeddedTextDiscriminator
from nalp.models.generators import LSTMGenerator

logger = l.get_logger(__name__)


class SeqGAN(Adversarial):
    """A SeqGAN class is the one in charge of Sequence Generative Adversarial Networks implementation.

    References:
        L. Yu, et al. Seqgan: Sequence generative adversarial nets with policy gradient.
        31th AAAI Conference on Artificial Intelligence (2017).

    """

    def __init__(self, encoder=None, vocab_size=1, max_length=1, embedding_size=32, hidden_size=64,
                 n_filters=(64), filters_size=(1), dropout_rate=0.25, temperature=1):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder for the generator.
            vocab_size (int): The size of the vocabulary for both discriminator and generator.
            max_length (int): Maximum length of the sequences for the discriminator.
            embedding_size (int): The size of the embedding layer for both discriminator and generator.
            hidden_size (int): The amount of hidden neurons for the generator.
            n_filters (tuple): Number of filters to be applied in the discriminator.
            filters_size (tuple): Size of filters to be applied in the discriminator.
            dropout_rate (float): Dropout activation rate.
            temperature (float): Temperature value to sample the token.

        """

        logger.info('Overriding class: Adversarial -> SeqGAN.')

        # Creating the discriminator network
        D = EmbeddedTextDiscriminator(
            vocab_size, max_length, embedding_size, n_filters, filters_size, dropout_rate)

        # Creating the generator network
        G = LSTMGenerator(encoder, vocab_size, embedding_size, hidden_size)

        super(SeqGAN, self).__init__(D, G, name='seqgan')

        # Defining a property for holding the vocabulary size
        self.vocab_size = vocab_size

        # Defining a property for holding the temperature
        self.T = temperature

        logger.info('Class overrided.')

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    @property
    def T(self):
        """float: Temperature value to sample the token.

        """

        return self._T

    @T.setter
    def T(self, T):
        self._T = T

    def compile(self, pre_optimizer, d_optimizer, g_optimizer):
        """Main building method.

        Args:
            pre_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the generator.
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.

        """

        # Creates optimizers for pre-training, discriminator and generator
        self.P_optimizer = pre_optimizer
        self.D_optimizer = d_optimizer
        self.G_optimizer = g_optimizer

        # Defining the loss function
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits

        # Defining both loss metrics
        self.D_loss = tf.metrics.Mean(name='D_loss')
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Storing losses as history keys
        self.history['pre_D_loss'] = []
        self.history['pre_G_loss'] = []
        self.history['D_loss'] = []
        self.history['G_loss'] = []

    def generate_batch(self, batch_size=1, length=1):
        """Generates a batch of tokens by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            batch_size (int): Size of the batch to be generated.
            length (int): Length of generated tokens.
            temperature (float): A temperature value to sample the token.

        Returns:
            A (batch_size, length) tensor of generated tokens.

        """

        # Generating an uniform tensor between 0 and vocab_size
        start_batch = tf.random.uniform(
            [batch_size, 1], 0, self.vocab_size, dtype='int32')

        # Copying the sampled batch with the start batch tokens
        sampled_batch = start_batch

        # Resetting the network states
        self.G.reset_states()

        # For every possible generation
        for _ in range(length):
            # Predicts the current token
            preds = self.G(start_batch)

            # Removes the second dimension of the tensor
            preds = tf.squeeze(preds, 1)

            # Regularize the prediction with the temperature
            preds /= self.T

            # Samples a predicted batch
            start_batch = tf.random.categorical(preds, 1, dtype='int32')

            # Concatenates the sampled batch with the predicted batch
            sampled_batch = tf.concat([sampled_batch, start_batch], 1)

        # Ignoring the last column to get the input sampled batch
        x_sampled_batch = sampled_batch[:, :length]

        # Ignoring the first column to get the input sampled batch
        y_sampled_batch = sampled_batch[:, 1:]

        return x_sampled_batch, y_sampled_batch

    def _get_reward(self, x, n_rollouts):
        """Calculates rewards over an input using a Monte Carlo search strategy.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            n_rollouts (int): Number of rollouts for conducting the Monte Carlo search.

        """

        # Gathers the batch size and maximum sequence length
        batch_size, max_length = x.shape[0], x.shape[1]

        # Creates an empty tensor for holding the rewards
        rewards = tf.zeros([1, batch_size])

        for _ in range(n_rollouts):
            # For every possible sequence step
            for step in range(1, max_length + 1):
                # Resetting the network states
                self.G.reset_states()

                # Calculate and gathers the last step output
                output = self.G(x)[:, -1, :]

                # Gathers the input upon to the current step
                samples = x[:, :step]

                # For every possible value ranging from step to maximum length
                for _ in range(step, max_length):
                    # Calculates the output
                    output = tf.random.categorical(output, 1, dtype='int32')

                    # Concatenates the samples with the output
                    samples = tf.concat([samples, output], 1)

                    # Squeezes the second dimension of the output tensor
                    output = tf.squeeze(self.G(output), 1)

                # Calculates the softmax over the discriminator output and removes the second dimension
                output = tf.squeeze(tf.math.softmax(self.D(samples)), 1)

                # Concatenates and accumulates the rewards tensor for every step
                rewards = tf.concat(
                    [rewards, tf.expand_dims(output[:, 1], 0)], 0)

        # Calculates the mean over the rewards tensor
        rewards = tf.reduce_mean(tf.reshape(
            rewards[1:, :], [batch_size, max_length, n_rollouts]), -1)

        return rewards

    @tf.function
    def G_pre_step(self, x, y):
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self.G(x)

            # Calculate the loss
            loss = tf.reduce_mean(self.loss(y, preds))

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.P_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    @tf.function
    def G_step(self, x, y, rewards):
        """Performs a single batch optimization step over the generator.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.
            rewards (tf.tensor): A tensor containing the rewards for the input.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self.G(x)

            # Calculate the loss
            loss = tf.reduce_mean(self.loss(y, preds) * rewards)

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    @tf.function
    def D_step(self, x, y):
        """Performs a single batch optimization step over the discriminator.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = tf.squeeze(self.D(x), 1)

            # Calculate the loss
            loss = tf.reduce_mean(self.loss(y, preds))

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.D.trainable_variables)

        # Apply gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(gradients, self.D.trainable_variables))

        # Updates the discriminator's loss state
        self.D_loss.update_state(loss)

    def pre_fit(self, batches, g_epochs=50, d_epochs=10):
        """Pre-trains the model.

        Args:
            batches (Dataset): Pre-training batches containing samples.
            g_epochs (int): The maximum number of pre-training generator epochs.
            d_epochs (int): The maximum number of pre-training discriminator epochs.

        """

        logger.info('Pre-fitting generator ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Iterate through all generator epochs
        for e in range(g_epochs):
            logger.info('Epoch %d/%d', e+1, g_epochs)

            # Resetting state to further append losses
            self.G_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)'])

            for x_batch, y_batch in batches:
                # Performs the optimization step over the generator
                self.G_pre_step(x_batch, y_batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result())])

            # Dump loss to history
            self.history['pre_G_loss'].append(self.D_loss.result().numpy())

            logger.to_file('Loss(G): %s', self.G_loss.result().numpy())

        logger.info('Pre-fitting discriminator ...')

        # Iterate through all discriminator epochs
        for e in range(d_epochs):
            logger.info('Epoch %d/%d', e+1, d_epochs)

            # Resetting state to further append losses
            self.D_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(D)'])

            for x_batch, _ in batches:
                # Gathering the batch size and the maximum sequence length
                batch_size, max_length = x_batch.shape[0], x_batch.shape[1]

                # Generates a batch of fake inputs
                x_fake_batch, _ = self.generate_batch(batch_size, max_length)

                # Concatenates real inputs and fake inputs into a single tensor
                x_concat_batch = tf.concat([x_batch, x_fake_batch], 0)

                # Creates a tensor holding label 0 for real samples and label 1 for fake samples
                y_concat_batch = tf.concat(
                    [tf.zeros(batch_size, dtype='int32'), tf.ones(batch_size, dtype='int32')], 0)

                # For a fixed amount of discriminator steps
                for _ in range(c.D_STEPS):
                    # Performs a random samples selection of batch size
                    indices = np.random.choice(
                        x_concat_batch.shape[0], batch_size, replace=False)

                    # Performs the optimization step over the discriminator
                    self.D_step(tf.gather(x_concat_batch, indices),
                                tf.gather(y_concat_batch, indices))

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(D)', self.D_loss.result())])

            # Dump loss to history
            self.history['pre_D_loss'].append(self.D_loss.result().numpy())

            logger.to_file('Loss(D): %s', self.D_loss.result().numpy())

    def fit(self, batches, epochs=10, g_epochs=1, d_epochs=5, n_rollouts=16):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of total training epochs.
            g_epochs (int): The maximum number of generator epochs per total epoch.
            d_epochs (int): The maximum number of discriminator epochs per total epoch.
            n_rollouts (int): Number of rollouts for conducting the Monte Carlo search.

        """

        logger.info('Fitting model ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for e in range(epochs):
            logger.info('Epoch %d/%d', e+1, epochs)

            # Resetting state to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)', 'loss(D)'])

            for x_batch, _ in batches:
                # Gathering the batch size and the maximum sequence length
                batch_size, max_length = x_batch.shape[0], x_batch.shape[1]

                # Iterate through all possible generator's  epochs
                for _ in range(g_epochs):
                    # Generates a batch of fake inputs
                    x_fake_batch, y_fake_batch = self.generate_batch(
                        batch_size, max_length)

                    # Gathers the rewards based on the sampled batch
                    rewards = self._get_reward(x_fake_batch, n_rollouts)

                    # Performs the optimization step over the generator
                    self.G_step(x_fake_batch, y_fake_batch, rewards)

                # Iterate through all possible discriminator's epochs
                for _ in range(d_epochs):
                    # Generates a batch of fake inputs
                    x_fake_batch, _ = self.generate_batch(
                        batch_size, max_length)

                    # Concatenates real inputs and fake inputs into a single tensor
                    x_concat_batch = tf.concat([x_batch, x_fake_batch], 0)

                    # Creates a tensor holding label 0 for real samples and label 1 for fake samples
                    y_concat_batch = tf.concat(
                        [tf.zeros(batch_size, dtype='int32'), tf.ones(batch_size, dtype='int32')], 0)

                    # For a fixed amount of discriminator steps
                    for _ in range(c.D_STEPS):
                        # Performs a random samples selection of batch size
                        indices = np.random.choice(
                            x_concat_batch.shape[0], batch_size, replace=False)

                        # Performs the optimization step over the discriminator
                        self.D_step(tf.gather(x_concat_batch, indices),
                                    tf.gather(y_concat_batch, indices))

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result()),
                                 ('loss(D)', self.D_loss.result())])

            # Dumps the losses to history
            self.history['G_loss'].append(self.G_loss.result().numpy())
            self.history['D_loss'].append(self.D_loss.result().numpy())

            logger.to_file('Loss(G): %s| Loss(D): %s', self.G_loss.result().numpy(), self.D_loss.result().numpy())
