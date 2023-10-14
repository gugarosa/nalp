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
        vocab_size: int = 1,
        embedding_size: int = 32,
        n_slots: int = 3,
        n_heads: int = 5,
        head_size: int = 10,
        n_blocks: int = 1,
        n_layers: int = 3,
        tau: float = 5,
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

        self.tau = tau

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

        x = self.embedding(x)
        x = self.rnn(x)
        x = self.linear(x)

        x_g, y_g = self.gumbel(x, self.tau)

        return x, x_g, y_g

    def generate_greedy_search(
        self, start: str, max_length: int = 100
    ) -> List[str]:
        """Generates text by using greedy search, where the sampled
        token is always sampled according to the maximum probability.

        Args:
            start: The start string to generate the text.
            max_length: Maximum length of generated text.

        Returns:
            (List[str]): Generated text.

        """

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_states()

        sampled_tokens = []
        for _ in range(max_length):
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            sampled_token = tf.argmax(preds, -1).numpy()

            start_tokens = tf.expand_dims(sampled_token, 0)

            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_temperature_sampling(
        self,
        start: str,
        max_length: int = 100,
        temperature: float = 1.0,
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

        self.tau = temperature

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_states()

        sampled_tokens = []
        for _ in range(max_length):
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            preds /= temperature

            sampled_token = tf.argmax(preds, -1).numpy()

            start_tokens = tf.expand_dims(sampled_token, 0)

            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_top_sampling(
        self,
        start: str,
        max_length: int = 100,
        k: int = 0,
        p: float = 0.0,
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

        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        self.reset_states()

        sampled_tokens = []
        for _ in range(max_length):
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            if k > 0:
                preds, preds_indexes = tf.math.top_k(preds, k)
            else:
                preds, preds_indexes = tf.math.top_k(preds, preds.shape[-1])

            if p > 0.0:
                cum_probs = tf.math.cumsum(tf.nn.softmax(preds), axis=-1)

                # Also ensures that first index will always be true to prevent zero
                # tokens from being sampled
                ignored_indexes = cum_probs <= p
                ignored_indexes = tf.tensor_scatter_nd_update(
                    ignored_indexes, [[0, 0]], [True]
                )

                preds = tf.expand_dims(preds[ignored_indexes], 0)
                preds_indexes = tf.expand_dims(preds_indexes[ignored_indexes], 0)

            # Samples the maximum top-k logit and gathers the real token index
            index = tf.argmax(preds, -1)[0]
            sampled_token = [preds_indexes[-1][index].numpy()]

            start_tokens = tf.expand_dims(sampled_token, 0)

            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            if sampled_token == c.EOS:
                break

        return sampled_tokens
