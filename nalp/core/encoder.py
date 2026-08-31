"""Encoder-related class."""

from typing import Any


class Encoder:
    """An Encoder class is responsible for receiving a Corpus and
    enconding it on a representation (i.e., integer, word2vec).

    """

    def __init__(self) -> None:
        """Initialization method."""

        self.encoder: Any = None

    def learn(self) -> None:
        """This method learns an encoding representation. Note that for each child,
        you need to define your own learning algorithm (representation).

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def encode(self) -> None:
        """This method encodes new data based on previous learning. Also, note that you
        need to define your own encoding algorithm when using its childs.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def decode(self) -> None:
        """This method decodes the encoded representation. Also, note that you
        need to define your own encoding algorithm when using its childs.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError
