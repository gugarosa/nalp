"""Encoder-related class.
"""

class Encoder:
    """An Encoder class is responsible for receiving a Corpus and
    enconding it on a representation (i.e., integer, word2vec).

    """

    @property
    def encoder(self):
        """obj: An encoder generic object.

        """

        return self._encoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    def learn(self):
        """This method learns an encoding representation. Note that for each child,
        you need to define your own learning algorithm (representation).

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def encode(self):
        """This method encodes new data based on previous learning. Also, note that you
        need to define your own encoding algorithm when using its childs.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def decode(self):
        """This method decodes the encoded representation. Also, note that you
        need to define your own encoding algorithm when using its childs.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError
