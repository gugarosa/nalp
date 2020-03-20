import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.models.base import AdversarialModel, Model
from nalp.models.recurrent.lstm import LSTM

logger = l.get_logger(__name__)


class Discriminator(Model):
    """A Discriminator class stands for the discriminative part of a Sequence Generative Adversarial Network.

    """

    def __init__(self, vocab_size, embedding_size):
        """Initialization method.

        Args:


        """

        logger.info('Overriding class: Model -> Discriminator.')

        # Overrides its parent class with any custom arguments if needed
        super(Discriminator, self).__init__(name='D_seqgan')

        # Creates an embedding layer
        self.embedding = layers.Embedding(
            vocab_size, embedding_size, name='embedding')

        #
        self.conv = layers.Conv2D(
            128, (3, 256), strides=(1, 1), padding='valid')

        #
        self.pool = layers.MaxPool1D(1, 128)

        #
        self.highway = layers.Dense(128)

        #
        self.out = layers.Dense(2)

        #
        self.drop = layers.Dropout(0.25)

    def call(self, x):
        # print(x.shape)

        x = self.embedding(x)

        x = tf.expand_dims(x, -1)

        # print(x.shape)

        x = tf.nn.relu(self.conv(x))

        # print(x.shape)

        x = tf.squeeze(x, 2)

        pool = self.pool(x)

        # print(pool.shape)

        x = self.highway(pool)

        # print(x.shape)

        x = tf.math.sigmoid(x) * tf.nn.relu(x) + \
            (1 - tf.math.sigmoid(x)) * pool

        x = self.out(self.drop(x))

        # print(x.shape)

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

    def __init__(self, encoder=None, vocab_size=1, embedding_size=1, hidden_size=1):
        """Initialization method.

        Args:


        """

        logger.info('Overriding class: AdversarialModel -> SeqGAN.')

        # Creating the discriminator network
        D = Discriminator(vocab_size, embedding_size)

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

    def pre_fit(self, batches, epochs=100, steps=3):
        """Pre-trains the model.

        Args:
            batches (Dataset): Pre-training batches containing samples.
            epochs (int): The maximum number of pre-training epochs.
            steps (int): Amount of pre-training steps per epoch for the discriminator.

        """

        logger.info('Pre-fitting generator ...')

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting state to further append losses
            self.G_loss.reset_states()

            # Iterate through all possible pre-training batches
            for x_batch, y_batch in batches:
                # Performs the optimization step over the generator
                self.G_pre_step(x_batch, y_batch)

            logger.info(f'Loss(G): {self.G_loss.result().numpy()}')

        logger.info('Pre-fitting discriminator ...')

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

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

                # For a fixed amount of steps
                for _ in range(steps):
                    # Performs a random samples selection of batch size
                    indices = np.random.choice(
                        x_batch.shape[0], batch_size, replace=False)

                    # Performs the optimization step over the discriminator
                    self.D_pre_step(tf.gather(x_batch, indices),
                                    tf.gather(y_batch, indices))

            logger.info(f'Loss(D): {self.D_loss.result().numpy()}')
