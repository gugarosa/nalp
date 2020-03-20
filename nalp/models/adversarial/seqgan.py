import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.models.base import AdversarialModel, Model
from nalp.models.recurrent.lstm import LSTM

logger = l.get_logger(__name__)


class Discriminator(Model):
    """A Discriminator class stands for the discriminative part of a Sequence Generative Adversarial Network.

    """

    def __init__(self, vocab_size, max_length, embedding_size, n_filters, filters_size, dropout_rate):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            max_length (int): Maximum length of the sequences.
            embedding_size (int): The size of the embedding layer.
            n_filters (list): Number of filters to be applied.
            filters_size (list): Size of filters to be applied.
            dropout_rate (float): Dropout activation rate.

        """

        logger.info('Overriding class: Model -> Discriminator.')

        # Overrides its parent class with any custom arguments if needed
        super(Discriminator, self).__init__(name='D_seqgan')

        # Creates an embedding layer
        self.embedding = layers.Embedding(
            vocab_size, embedding_size, name='embedding')

        # Defining a list for holding the convolutional layers
        self.conv = [layers.Conv2D(n, (k, embedding_size), strides=(
            1, 1), padding='valid') for n, k in zip(n_filters, filters_size)]

        # Defining a list for holding the pooling layers
        self.pool = [layers.MaxPool1D(max_length - k + 1, 1)
                     for k in filters_size]

        # Defining a linear layer for serving as the `highway`
        self.highway = layers.Dense(sum(n_filters))

        # Defining the dropout layer
        self.drop = layers.Dropout(dropout_rate)

        # And finally, defining the output layer
        self.out = layers.Dense(2)

    def call(self, x):
        # Passing down the embedding layer
        x = self.embedding(x)

        # Expanding the last dimension
        x = tf.expand_dims(x, -1)

        # Passing down the convolutional layers following a ReLU activation
        # and removal of third dimension
        convs = [tf.squeeze(tf.nn.relu(conv(x)), 2) for conv in self.conv]

        # Passing down the pooling layers per convolutional layer
        pools = [pool(conv) for pool, conv in zip(self.pool, convs)]

        # Concatenating all the pooling outputs into a single tensor
        x = tf.concat(pools, axis=2)

        # Calculating the output of the linear layer
        hw = self.highway(x)

        # Calculating the `highway` layer
        x = tf.math.sigmoid(hw) * tf.nn.relu(hw) + (1 - tf.math.sigmoid(hw)) * x

        # Calculating the output with a dropout regularization
        x = self.out(self.drop(x))

        return x


class Generator(LSTM):
    """A Generator class stands for the generator part of a Sequence Generative Adversarial Network.

    """

    def __init__(self, encoder, vocab_size, embedding_size, hidden_size):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder.
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding layer.
            hidden_size (int): The amount of hidden neurons.

        """

        logger.info('Overriding class: LSTM -> Generator.')

        # Overrides its parent class with any custom arguments if needed
        super(Generator, self).__init__(encoder=encoder, vocab_size=vocab_size,
                                        embedding_size=embedding_size, hidden_size=hidden_size)

    def generate_batch(self, batch_size=1, length=1, temperature=1.0):
        """Generates a batch of tokens by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            batch_size (int): Size of the batch to be generated.
            length (int): Length of generated tokens.
            temperature (float): A temperature value to sample the token.

        Returns:
            A (batch_size, length) tensor of generated tokens.

        """

        # Creating an empty tensor for the starting batch
        start_batch = tf.zeros([batch_size, 1])

        # Creating an empty tensor for the sampled batch
        sampled_batch = tf.zeros([batch_size, 1], dtype='int64')

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for i in range(length):
            # Predicts the current token
            preds = self(start_batch)

            # Removes the second dimension of the tensor
            preds = tf.squeeze(preds, 1)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a predicted batch
            start_batch = tf.random.categorical(preds, num_samples=1)

            # Concatenates the sampled batch with the predicted batch
            sampled_batch = tf.concat([sampled_batch, start_batch], axis=1)

        # Ignores the first column of the sampled batch
        sampled_batch = sampled_batch[:, 1:]

        return sampled_batch


class SeqGAN(AdversarialModel):
    """A SeqGAN class is the one in charge of Sequence Generative Adversarial Networks implementation.

    References:
        L. Yu, et al. Seqgan: Sequence generative adversarial nets with policy gradient. 31th AAAI Conference on Artificial Intelligence (2017).

    """

    def __init__(self, encoder=None, vocab_size=1, max_length=1, embedding_size=1, hidden_size=1, n_filters=[64], filters_size=[1], dropout_rate=0.25):
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

        """

        logger.info('Overriding class: AdversarialModel -> SeqGAN.')

        # Creating the discriminator network
        D = Discriminator(vocab_size, max_length, embedding_size,
                          n_filters, filters_size, dropout_rate)

        # Creating the generator network
        G = Generator(encoder, vocab_size, embedding_size, hidden_size)

        # Overrides its parent class with any custom arguments if needed
        super(SeqGAN, self).__init__(D, G, name='seqgan')

    @tf.function
    def G_pre_step(self, x, y):
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            x (tf.Tensor): A tensor containing the inputs.
            y (tf.Tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self.G(x)

            # Calculate the loss
            loss = self.loss(y, preds)

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    @tf.function
    def D_pre_step(self, x, y):
        """Performs a single batch optimization pre-fitting step over the discriminator.

        Args:
            x (tf.Tensor): A tensor containing the inputs.
            y (tf.Tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self.D(x)

            # Calculate the loss
            loss = self.loss(y, preds)

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.D.trainable_variables)

        # Apply gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(gradients, self.D.trainable_variables))

        # Updates the discriminator's loss state
        self.D_loss.update_state(loss)

    def pre_fit(self, batches, g_epochs=100, d_epochs=10, d_steps=3):
        """Pre-trains the model.

        Args:
            batches (Dataset): Pre-training batches containing samples.
            g_epochs (int): The maximum number of pre-training generator epochs.
            d_epochs (int): The maximum number of pre-training discriminator epochs.
            d_steps (int): Amount of pre-training steps per epoch for the discriminator.

        """

        logger.info('Pre-fitting generator ...')

        # Iterate through all generator epochs
        for e in range(g_epochs):
            logger.info(f'Epoch {e+1}/{g_epochs}')

            # Resetting state to further append losses
            self.G_loss.reset_states()

            # Iterate through all possible pre-training batches
            for x_batch, y_batch in batches:
                # Performs the optimization step over the generator
                self.G_pre_step(x_batch, y_batch)

            logger.info(f'Loss(G): {self.G_loss.result().numpy()}')

        logger.info('Pre-fitting discriminator ...')

        # Iterate through all discriminator epochs
        for e in range(d_epochs):
            logger.info(f'Epoch {e+1}/{d_epochs}')

            # Resetting state to further append losses
            self.D_loss.reset_states()

            # Iterate through all possible pre-training batches
            for x_batch, _ in batches:
                # Gathering the batch size and the maximum sequence length
                batch_size, max_length = x_batch.shape[0], x_batch.shape[1]

                # Generates a batch of fake inputs
                x_fake_batch = self.G.generate_batch(
                    batch_size, max_length, 0.5)

                # Concatenates real inputs and fake inputs into a single tensor
                x_batch = tf.concat([x_batch, x_fake_batch], axis=0)

                # Creates a tensor holding label 0 for real samples and label 1 for fake samples
                y_batch = tf.concat(
                    [tf.zeros(batch_size,), tf.ones(batch_size,)], axis=0)

                # For a fixed amount of discriminator steps
                for _ in range(d_steps):
                    # Performs a random samples selection of batch size
                    indices = np.random.choice(
                        x_batch.shape[0], batch_size, replace=False)

                    # Performs the optimization step over the discriminator
                    self.D_pre_step(tf.gather(x_batch, indices),
                                    tf.gather(y_batch, indices))

            logger.info(f'Loss(D): {self.D_loss.result().numpy()}')
