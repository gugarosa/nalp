import tensorflow as tf
from tensorflow.keras import Model

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class SimpleNeural(Model):
    """A SimpleNeural class is responsible for easily-implementing a neural network, when
    custom training or additional sets are not needed.

    """

    def __init__(self, name):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(SimpleNeural, self).__init__(name=name)

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def generate_text(self, encoder, start, length=100, temperature=1.0):
        """Generates text by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder.
            start (str): The start string to generate the text.
            length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Returns:
            A list of generated text.

        """

        logger.debug(f'Generating text with length: {length} ...')

        # Encoding the start string into tokens
        start_tokens = encoder.encode(start)

        # Expanding the first dimension of tensor
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for i in range(length):
            # Predicts the current token
            preds = self(start_tokens)

            # Removes the first dimension of the tensor
            preds = tf.squeeze(preds, 0)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a predicted token
            sampled_token = tf.random.categorical(
                preds, num_samples=1)[-1, 0].numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims([sampled_token], 0)

            # Appends the sampled token to the list
            sampled_tokens.append(sampled_token)

        # Decodes the list into raw text
        text = encoder.decode(sampled_tokens)

        return text
