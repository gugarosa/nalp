import tensorflow as tf

import nalp.utils.logging as l
import nalp.utils.math as m
from nalp.models.generators.lstm import LSTMGenerator

logger = l.get_logger(__name__)


class GumbelLSTMGenerator(LSTMGenerator):
    """A GumbelLSTMGenerator class is the one in charge of a generative Gumbel-based Long Short-Term Memory implementation.

    References:
        E. Jang, S. Gu, B. Poole. Categorical reparameterization with gumbel-softmax. Preprint arXiv:1611.01144 (2016).

    """

    def __init__(self, encoder=None, vocab_size=1, embedding_size=32, hidden_size=64, tau=5):
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
            Softmax outputs, predicted token and logit-based predictions.

        """

        # Firstly, we apply the embedding layer
        x = self.embedding(x)

        # We need to apply the input into the first recurrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        # Adding a sampled Gumbel distribution to the output
        x_g = x + m.gumbel_distribution(x.shape)

        # Sampling an argmax token from the Gumbel-based output
        y_g = tf.stop_gradient(tf.argmax(x_g, -1))

        # Applying the softmax over the Gumbel-based output
        x_g = tf.nn.softmax(x_g * self.tau)

        return x_g, y_g, x

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

        logger.debug(f'Generating text with length: {length} ...')

        # Applying Gumbel-Softmax temperature as argument
        self.G.tau = temperature

        # Encoding the start string into tokens
        start_tokens = self.encoder.encode(start)

        # Expanding the first dimension of tensor
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for i in range(length):
            # Predicts the current token
            preds, _, _ = self(start_tokens)

            # Removes the first dimension of the tensor
            preds = tf.squeeze(preds, 0)

            # Samples a predicted token
            sampled_token = tf.argmax(preds, -1)[-1].numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims([sampled_token], 0)

            # Appends the sampled token to the list
            sampled_tokens.append(sampled_token)

        # Decodes the list into raw text
        text = self.encoder.decode(sampled_tokens)

        return text
