class Corpus():
    """A Corpus class is used to defined the first step of the workflow.

    It serves as a basis class to load raw text, documents (list of sentences) and audio.

    """

    def __init__(self):
        """Initialization method.

        """

        # Creates a tokens property
        self.tokens = None

    @property
    def tokens(self):
        """list: A list of tokens.

        """

        return self._tokens

    @tokens.setter
    def tokens(self, tokens):
        self._tokens = tokens

    def _build(self):
        """This method serves to build up the Corpus class. Note that for each child,
        you need to define your own building method.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError
