class Encoder:
    """An Encoder class is responsible for receiving raw data and
    enconding it on a representation (i.e., count vectorizer, tfidf, word2vec).

    """

    def __init__(self):
        """Initialization method.

        Some basic shared variables between Encoder's childs should be declared here.

        """

        # The encoder object will be initialized as None
        self._encoder = None

        # We also set the encoded data as None
        self._encoded_data = None

    @property
    def encoder(self):
        """obj: An encoder generic object.

        """

        return self._encoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @property
    def encoded_data(self):
        """np.array: A numpy array holding the encoded data.

        """

        return self._encoded_data

    @encoded_data.setter
    def encoded_data(self, encoded_data):
        self._encoded_data = encoded_data

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
