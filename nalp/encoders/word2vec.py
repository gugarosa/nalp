# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Word2vec encoding."""

import multiprocessing

import numpy as np
from gensim.models.word2vec import Word2Vec as W2V

from nalp.core.encoder import Encoder


class Word2vecEncoder(Encoder):
    """Learn Word2Vec vectors and decode vectors through vocabulary similarity."""

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
        """Train a Word2Vec representation for a token sequence.

        Replace the current Gensim model and treat the tokens as one training sentence.
        Training uses the available CPU count for worker parallelism.

        Args:
            tokens: Token sequence used to learn the vocabulary and vectors.
            max_features: Number of dimensions in each word vector.
            window_size: Maximum distance between a word and its context.
            min_count: Minimum token count required for inclusion in the vocabulary.
            algorithm: Training algorithm, with 0 selecting CBOW and 1 selecting skip-gram.
            learning_rate: Initial training learning rate.
            iterations: Number of training epochs.

        Raises:
            RuntimeError: The corpus does not yield a trainable vocabulary.

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
        """Look up learned word vectors for a token sequence.

        Args:
            tokens: Vocabulary tokens whose vectors should be returned.

        Returns:
            A float64 array shaped as the token count by the learned vector dimension.

        Raises:
            RuntimeError: The encoder is None because no model has been learned.
            KeyError: A token is absent from the learned vocabulary.

        """

        if self.encoder is None:
            raise RuntimeError("`encoder` is None, call learn() before encode().")

        wv = self.encoder.wv

        encoded_tokens = np.zeros((len(tokens), self.encoder.vector_size))
        for i, token in enumerate(tokens):
            encoded_tokens[i, :] = wv[token]

        return encoded_tokens

    def decode(self, encoded_tokens: np.ndarray) -> list[str]:
        """Find the most similar learned token for each input vector.

        Args:
            encoded_tokens: Vectors whose final dimension matches the learned vector dimension.

        Returns:
            The most similar vocabulary token for each vector.

        Raises:
            RuntimeError: The encoder is None because no model has been learned.

        """

        if self.encoder is None:
            raise RuntimeError("`encoder` is None, call learn() before decode().")

        decoded_tokens = [self.encoder.wv.most_similar(positive=[t])[0][0] for t in encoded_tokens]

        return decoded_tokens
