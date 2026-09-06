# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Encoder-related class."""

from typing import Any


class Encoder:
    """Define the learning, encoding, and decoding interface for token representations."""

    def __init__(self) -> None:
        """Initialize an encoder without a learned representation."""

        self.encoder: Any = None

    def learn(self) -> None:
        """Learn the representation implemented by a concrete encoder.

        Raises:
            NotImplementedError: The concrete encoder does not implement learning.

        """

        raise NotImplementedError("`Encoder.learn` must be implemented by a subclass.")

    def encode(self) -> None:
        """Encode tokens using a concrete representation.

        Raises:
            NotImplementedError: The concrete encoder does not implement encoding.

        """

        raise NotImplementedError("`Encoder.encode` must be implemented by a subclass.")

    def decode(self) -> None:
        """Decode a concrete representation back to tokens.

        Raises:
            NotImplementedError: The concrete encoder does not implement decoding.

        """

        raise NotImplementedError("`Encoder.decode` must be implemented by a subclass.")
