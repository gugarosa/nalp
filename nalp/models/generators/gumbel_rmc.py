"""Gumbel Relational Memory Core generator.
"""

from typing import List, Optional, Tuple

import tensorflow as tf

import nalp.utils.constants as c
from nalp.encoders.integer import IntegerEncoder
from nalp.models.generators import RMCGenerator
from nalp.models.layers import GumbelSoftmax
from nalp.utils import logging

logger = logging.get_logger(__name__)


class GumbelRMCGenerator(RMCGenerator):
    """A GumbelRMCGenerator class is the one in charge of a
    generative Gumbel-based Relational Memory Core implementation.

    """

    def __init__(
        self,
        encoder: Optional[IntegerEncoder] = None,
        vocab_size: Optional[int] = 1,
        embedding_size: Optional[int] = 32,
        n_slots: Optional[int] = 3,
        n_heads: Optional[int] = 5,
        head_size: Optional[int] = 10,
        n_blocks: Optional[int] = 1,
        n_layers: Optional[int] = 3,
        tau: Optional[float] = 5,
    ):
        """Initialization method.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            n_slots: Number of memory slots.
            n_heads: Number of attention heads.
            head_size: Size of each attention head.
            n_blocks: Number of feed-forward networks.
            n_layers: Amout of layers per feed-forward network.
            tau: Gumbel-Softmax temperature parameter.

        """

        logger.info("Overriding class: RMCGenerator -> GumbelRMCGenerator.")

        super(GumbelRMCGenerator, self).__init__(
            encoder,
            vocab_size,
            embedding_size,
            n_slots,
            n_heads,
            head_size,
            n_blocks,
            n_layers,
        )

        # Defining a property to hold the Gumbel-Softmax temperature parameter
        self.tau = tau

        # Creates a Gumbel-Softmax layer
        self.gumbel = GumbelSoftmax(name="gumbel")

        logger.info("Class overrided.")

    @property
    def tau(self) -> float:
        """Gumbel-Softmax temperature parameter."""

        return self._tau

    @tau.setter
    def tau(self, tau: float) -> None:
        self._tau = tau

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Logit-based predictions, Gumbel-Softmax outputs and predicted token.

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

    def generate_greedy_search(
        self, start: str, max_length: Optional[int] = 100
    ) -> List[str]:
        """Generates text by using greedy search, where the sampled
        token is always sampled according to the maximum probability.

        Args:
            start: The start string to generate the text.
            max_length: Maximum length of generated text.

        Returns:
            (List[str]): Generated text.

        """

        # Encoding the start string into tokens and expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers its last timestep
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            # Samples a predicted token
            sampled_token = tf.argmax(preds, -1).numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_temperature_sampling(
        self,
        start: str,
        max_length: Optional[int] = 100,
        temperature: Optional[float] = 1.0,
    ):
        """Generates text by using temperature sampling, where the sampled
        token is sampled according to a multinomial/categorical distribution.

        Args:
            start: The start string to generate the text.
            max_length: Length of generated text.
            temperature: A temperature value to sample the token.

        Returns:
            (List[str]): Generated text.

        """

        # Applying Gumbel-Softmax temperature as argument
        self.tau = temperature

        # Encoding the start string into tokens and expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers its last timestep
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a predicted token
            sampled_token = tf.argmax(preds, -1).numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_top_sampling(
        self,
        start: str,
        max_length: Optional[int] = 100,
        k: Optional[int] = 0,
        p: Optional[float] = 0.0,
    ):
        """Generates text by using top-k and top-p sampling, where the sampled
        token is sampled according to the `k` most likely words distribution, as well
        as to the maximim cumulative probability `p`.

        Args:
            start: The start string to generate the text.
            max_length: Length of generated text.
            k: Indicates the amount of likely words.
            p: Maximum cumulative probability to be thresholded.

        Returns:
            (List[str]): Generated text.

        """

        # Encoding the start string into tokens and expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers its last timestep
            _, preds, _ = self(start_tokens)
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
                ignored_indexes = tf.tensor_scatter_nd_update(
                    ignored_indexes, [[0, 0]], [True]
                )

                # Filters the predictions and its indexes
                preds = tf.expand_dims(preds[ignored_indexes], 0)
                preds_indexes = tf.expand_dims(preds_indexes[ignored_indexes], 0)

            # Samples the maximum top-k logit and gathers the real token index
            index = tf.argmax(preds, -1)[0]
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
