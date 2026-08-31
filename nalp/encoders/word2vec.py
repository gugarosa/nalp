"""Word2vec encoding."""

import multiprocessing

import numpy as np
from gensim.models.word2vec import Word2Vec as W2V

from nalp.core.encoder import Encoder


class Word2vecEncoder(Encoder):
    """A Word2vecEncoder class is responsible for learning a Word2Vec encode and
    further encoding new data.

    """

    def learn(
        self,
        tokens: list[str],
        max_features: int = 128,
        window_size: int = 5,
        min_count: int = 1,
        algorithm: int = 0,
        learning_rate: float = 0.01,
        iterations: int = 1000,
    ) -> None:
        """Learns a Word2Vec representation based on the its methodology.

        One can use CBOW or Skip-gram algorithm for the learning procedure.

        Args:
            tokens: A list of tokens.
            max_features: Maximum number of features to be fitted.
            window_size: Maximum distance between current and predicted word.
            min_count: Minimum count of words for its use.
            algorithm: 1 for skip-gram, while 0 for CBOW.
            learning_rate: Value of the learning rate.
            iterations: Number of iterations.

        """

        self.encoder = W2V(
            sentences=[tokens],
            vector_size=max_features,
            window=window_size,
            min_count=min_count,
            sg=algorithm,
            alpha=learning_rate,
            epochs=iterations,
            workers=multiprocessing.cpu_count(),
        )

    def encode(self, tokens: list[str]) -> np.ndarray:
        """Encodes the data into a Word2Vec representation.

        Args:
            tokens: Tokens to be encoded.

        """

        if self.encoder is None:
            raise RuntimeError("You need to call learn() prior to encode() method.")

        wv = self.encoder.wv

        encoded_tokens = np.zeros((len(tokens), self.encoder.vector_size))
        for i, token in enumerate(tokens):
            encoded_tokens[i, :] = wv[token]

        return encoded_tokens

    def decode(self, encoded_tokens: np.ndarray) -> list[str]:
        """Decodes the encoding back to tokens.

        Args:
            encoded_tokens: A numpy array containing the encoded tokens.

        Returns:
            (List[str]): Decoded tokens.

        """

        if self.encoder is None:
            raise RuntimeError("You need to call learn() prior to decode() method.")

        decoded_tokens = [
            self.encoder.wv.most_similar(positive=[t])[0][0] for t in encoded_tokens
        ]

        return decoded_tokens
