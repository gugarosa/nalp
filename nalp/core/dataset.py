import numpy as np


class Dataset:
    """A Dataset class is responsible for receiving raw tokens (words or chars) and
    creating properties that will be feed as an input to the networks (i.e., vocabulary and indexes).

    Properties:
        tokens (list): A list holding tokenized words or characters.
        vocab_size (int): The size of the vocabulary.
        vocab_index (dict): A dictionary mapping vocabulary to indexes.
        index_vocab (dict): A dictionary mapping indexes to vocabulary.
        tokens_idx (np.array): A numpy array holding the indexed tokens.

    Methods:
        vocab_to_index(vocab): Maps a vocabulary to integer indexes.
        index_to_vocab(vocab): Maps integer indexes to a vocabulary.
        indexate_tokens(tokens, vocab_index): Indexates tokens based on a previous defined vocabulary.

    """

    def __init__(self, tokens):
        """Initialization method.
        Some basic shared variables and methods between Datasets's childs
        should be declared here.

        Args:
            tokens (list): A list holding tokenized words or characters.

        """

        # Firstly, we need to define a tokens property
        self._tokens = tokens

        # Calculates the vocabulary and its size from tokens
        vocab = list(set(tokens))
        self._vocab_size = len(vocab)

        # Creates a dictionary mapping vocabulary to indexes
        self._vocab_index = self.vocab_to_index(vocab)

        # Creates a dictionary mapping indexes to vocabulary
        self._index_vocab = self.index_to_vocab(vocab)

        # Indexate tokens based on a vocabulary-index dictionary
        self._tokens_idx = self.indexate_tokens(tokens, self._vocab_index)

    @property
    def tokens(self):
        """A list holding tokenized words or characters.

        """

        return self._tokens

    @property
    def vocab_size(self):
        """The size of the vocabulary.

        """

        return self._vocab_size

    @property
    def vocab_index(self):
        """A dictionary mapping vocabulary to indexes.

        """

        return self._vocab_index

    @property
    def index_vocab(self):
        """A dictionary mapping indexes to vocabulary.

        """

        return self._index_vocab

    @property
    def tokens_idx(self):
        """A numpy array holding the indexed tokens.

        """

        return self._tokens_idx

    def vocab_to_index(self, vocab):
        """Maps a vocabulary to integer indexes.

        Args:
            vocab (list): A list holding the vocabulary.

        """

        vocab_to_index = {c: i for i, c in enumerate(vocab)}
        return vocab_to_index

    def index_to_vocab(self, vocab):
        """Maps integer indexes to a vocabulary.

        Args:
            vocab (list): A list holding the vocabulary.

        """

        index_to_vocab = {i: c for i, c in enumerate(vocab)}
        return index_to_vocab

    def indexate_tokens(self, tokens, vocab_index):
        """Indexates tokens based on a previous defined vocabulary.

        Args:
            tokens (list): A list holding tokenized words or characters.
            vocab_index (dict): A dictionary mapping vocabulary to indexes.

        """

        tokens_idx = np.array([vocab_index[c] for c in tokens])
        return tokens_idx
