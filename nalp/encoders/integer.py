"""Integer-based encoding.
"""

import numpy as np

import nalp.utils.constants as c
from nalp.core.encoder import Encoder


class IntegerEncoder(Encoder):
    """An IntegerEncoder class is responsible for encoding text into integers."""

    def __init__(self) -> None:
        """Initialization method."""

        super().__init__()
        self.decoder: dict[int, str] | None = None

    def learn(
        self, dictionary: dict[str, int], reverse_dictionary: dict[int, str]
    ) -> None:
        """Learns an integer vectorization encoding.

        Args:
            dictionary: The vocabulary to index mapping.
            reverse_dictionary: The index to vocabulary mapping.

        """

        self.encoder = dictionary
        self.decoder = reverse_dictionary

    def encode(self, tokens: list[str] | list[list[str]]) -> np.ndarray:
        """Encodes new tokens based on previous learning.

        Args:
            tokens: A list of tokens to be encoded.

        Returns:
            (np.array): Encoded tokens.

        """

        if self.encoder is None:
            raise RuntimeError("You need to call learn() prior to encode() method.")

        encoded_tokens = []

        for token in tokens:
            if isinstance(token, (np.ndarray, list)):
                encoded_tokens.append(
                    [
                        self.encoder[t] if t in self.encoder else self.encoder[c.UNK]
                        for t in token
                    ]
                )

            else:
                if token in self.encoder:
                    encoded_tokens.append(self.encoder[token])

                else:
                    encoded_tokens.append(self.encoder[c.UNK])

        encoded_tokens = np.array(encoded_tokens, dtype=np.int32)

        return encoded_tokens

    def decode(self, encoded_tokens: np.ndarray) -> list[str] | list[list[str]]:
        """Decodes the encoding back to tokens.

        Args:
            encoded_tokens: A numpy array containing the encoded tokens.

        Returns:
            (List[str]): Decoded tokens.

        """

        if self.decoder is None:
            raise RuntimeError("You need to call learn() prior to decode() method.")

        decoded_tokens = []

        for token in encoded_tokens:
            if isinstance(token, (np.ndarray, list)):
                decoded_tokens.append([self.decoder[t] for t in token])

            else:
                decoded_tokens.append(self.decoder[token])

        return decoded_tokens
