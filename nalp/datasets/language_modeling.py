# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Language modeling dataset class."""

import numpy as np
import tensorflow as tf

from nalp.core.dataset import Dataset


class LanguageModelingDataset(Dataset):
    """Create paired input and next-token target batches."""

    def __init__(
        self,
        encoded_tokens: np.ndarray,
        max_contiguous_pad_length: int = 1,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> None:
        """Create shifted token sequences and batch them for language modeling.

        Flat input is grouped into nonoverlapping windows of input length plus one.
        Rectangular sentence input is already grouped and is shifted along its final dimension.
        Incomplete windows and final batches are discarded without modifying the input array.

        Args:
            encoded_tokens: Integer token IDs as a flat sequence or a rectangular array of sentences.
            max_contiguous_pad_length: Input timesteps per window when tokens are one-dimensional.
            batch_size: Number of input-target sequence pairs in each complete batch.
            shuffle: Whether to shuffle sequence pairs before batching.

        """

        super().__init__(shuffle)

        sequences = self._create_sequences(encoded_tokens, max_contiguous_pad_length)
        mapped_sequences = sequences.map(self._create_input_target)

        self._build(mapped_sequences, batch_size)

    def _create_sequences(self, encoded_tokens: np.ndarray, max_contiguous_pad_length: int) -> tf.data.Dataset:
        sequences = tf.data.Dataset.from_tensor_slices(encoded_tokens)

        # Sentence arrays already carry sequence boundaries
        if encoded_tokens.ndim == 1:
            sequences = sequences.batch(max_contiguous_pad_length + 1, drop_remainder=True)

        return sequences

    def _create_input_target(self, sequence: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return sequence[:-1], sequence[1:]
