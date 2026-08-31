"""Language modeling dataset class."""

import numpy as np
import tensorflow as tf

from nalp.core import Dataset


class LanguageModelingDataset(Dataset):
    """A LanguageModelingDataset class is responsible for creating a dataset
    that predicts the next timestep (t+1) given a timestep (t).

    """

    def __init__(
        self,
        encoded_tokens: np.ndarray,
        max_contiguous_pad_length: int = 1,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> None:
        """Initialization method.

        Args:
            encoded_tokens: An array of encoded tokens.
            max_contiguous_pad_length: Maximum length to pad contiguous text.
            batch_size: Size of batches.
            shuffle: Whether batches should be shuffled or not.

        """

        super().__init__(shuffle)

        sequences = self._create_sequences(encoded_tokens, max_contiguous_pad_length)
        mapped_sequences = sequences.map(self._create_input_target)

        self._build(mapped_sequences, batch_size)

    def _create_sequences(
        self, encoded_tokens: np.ndarray, max_contiguous_pad_length: int
    ) -> tf.data.Dataset:
        """Creates sequences of the desired length.

        Args:
            encoded_tokens: An array of encoded tokens.
            max_contiguous_pad_length: Maximum sequences' length.

        Returns:
            (tf.data.Dataset): Slices of tensor-based sequences.

        """

        sequences = tf.data.Dataset.from_tensor_slices(encoded_tokens)

        # This means that is a contiguous sequence of tokens and needs to
        # be parsed into individual sequences
        if encoded_tokens.ndim == 1:
            sequences = sequences.batch(
                max_contiguous_pad_length + 1, drop_remainder=True
            )

        return sequences

    def _create_input_target(self, sequence: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Creates input (t) and targets (t+1) using the next timestep approach.

        Args:
            sequence: A tensor holding the sequence to be mapped.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor]): Input and target tensors.

        """

        return sequence[:-1], sequence[1:]
