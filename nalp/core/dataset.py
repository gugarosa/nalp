class Dataset:
    """A Dataset class is responsible for receiving encoded tokens and
    persisting data that will be feed as an input to the networks.

    """

    def __init__(self, encoded_tokens):
        """Initialization method.

        Args:
            encoded_tokens (np.array): An array of encoded tokens.

        """

        # Creating a property to hold the encoded tokens
        self.encoded_tokens = encoded_tokens

        # Creating a property to hold the further batches
        self.batches = None

    @property
    def encoded_tokens(self):
        """np.array: An numpy array holding the encoded tokens.

        """

        return self._encoded_tokens

    @encoded_tokens.setter
    def encoded_tokens(self, encoded_tokens):
        self._encoded_tokens = encoded_tokens

    @property
    def batches(self):
        """tf.data.Dataset: An instance of tensorflow's dataset batches.

        """

        return self._batches

    @batches.setter
    def batches(self, batches):
        self._batches = batches

    def _build(self):
        """This method serves to build up the Dataset class. Note that for each child,
        you need to define your own building method.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError
