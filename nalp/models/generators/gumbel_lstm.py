import tensorflow as tf

import nalp.utils.logging as l
from nalp.models.generators.lstm import LSTMGenerator
from nalp.models.layers.gumbel_softmax import GumbelSoftmax

logger = l.get_logger(__name__)


class GumbelLSTMGenerator(LSTMGenerator):
    """A GumbelLSTMGenerator class is the one in charge of a generative Gumbel-based Long Short-Term Memory implementation.

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

        # Creates a Gumbel-Softmax layer
        self.gumbel = GumbelSoftmax(name='gumbel')

    @property
    def tau(self):
        """float: Gumbel-Softmax temperature parameter.

        """

        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            Logit-based predictions, Gumbel-Softmax outputs and predicted token.

        """

        # Firstly, we apply the embedding layer
        x = self.embedding(x)

        # We need to apply the input into the first recurrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        # Lastly, we apply the Gumbel-Softmax layer
        x_g, y_g = self.gumbel(x, self.tau)

        return x, x_g, y_g

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
        self.tau = temperature

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
            _, preds, _ = self(start_tokens)

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
