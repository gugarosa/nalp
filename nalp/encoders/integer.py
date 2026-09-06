# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Integer-based encoding."""

import numpy as np

import nalp.utils.constants as c
from nalp.core.encoder import Encoder


class IntegerEncoder(Encoder):
    """Translate tokens and integer IDs through learned vocabulary mappings."""

    def __init__(self) -> None:
        """Initialize an integer encoder without vocabulary mappings."""

        super().__init__()
        self.decoder: dict[int, str] | None = None

    def learn(self, dictionary: dict[str, int], reverse_dictionary: dict[int, str]) -> None:
        """Bind token-to-ID and ID-to-token mappings without copying them.

        Caller mutations to either dictionary affect subsequent encoding and decoding.

        Args:
            dictionary: Token-to-integer mapping used for encoding.
            reverse_dictionary: Integer-to-token mapping used for decoding.

        """

        self.encoder = dictionary
        self.decoder = reverse_dictionary

    def encode(self, tokens: str | list[str] | list[list[str]] | np.ndarray) -> np.ndarray:
        """Encode tokens as int32 vocabulary IDs.

        Unknown tokens use the learned unknown-token entry, and input tokens are not modified.

        Args:
            tokens: Character string, flat tokens, or a rectangular collection of token lists.

        Returns:
            An int32 array preserving the flat or rectangular nested shape of the tokens.

        Raises:
            RuntimeError: The encoder is None because no mapping has been learned.
            KeyError: An unknown token is encountered without an unknown-token mapping.
            ValueError: Nested token lists do not form a rectangular array.

        """

        if self.encoder is None:
            raise RuntimeError("`encoder` is None, call learn() before encode().")

        encoded_tokens = []

        for token in tokens:
            if isinstance(token, (np.ndarray, list)):
                encoded_tokens.append([self.encoder[t] if t in self.encoder else self.encoder[c.UNK] for t in token])

            else:
                if token in self.encoder:
                    encoded_tokens.append(self.encoder[token])

                else:
                    encoded_tokens.append(self.encoder[c.UNK])

        encoded_tokens = np.array(encoded_tokens, dtype=np.int32)

        return encoded_tokens

    def decode(self, encoded_tokens: np.ndarray | list[int] | list[list[int]]) -> list[str] | list[list[str]]:
        """Decode vocabulary IDs into flat or nested token lists.

        Args:
            encoded_tokens: Flat or rectangular nested integer IDs.

        Returns:
            Token strings with the corresponding flat or nested list structure.

        Raises:
            RuntimeError: The decoder is None because no mapping has been learned.
            KeyError: An input ID is absent from the reverse mapping.

        """

        if self.decoder is None:
            raise RuntimeError("`decoder` is None, call learn() before decode().")

        decoded_tokens = []

        for token in encoded_tokens:
            if isinstance(token, (np.ndarray, list)):
                decoded_tokens.append([self.decoder[t] for t in token])

            else:
                decoded_tokens.append(self.decoder[token])

        return decoded_tokens
