import numpy as np


class Dataset:
    """A Dataset class is responsible for receiving raw tokens (words or chars) and
    creating properties that will be feed as an input to the networks (i.e., vocabulary and indexes).

    """

    def __init__(self, tokens=None):
        """Initialization method.
        Some basic shared variables and methods between Datasets's childs
        should be declared here.

        Args:
            tokens (list): A list holding tokenized words or characters.

        """

        # List of tokens
        self._tokens = None

        # The size of the vocabulary
        self._vocab_size = None

        # A dictionary mapping vocabulary to indexes
        self._vocab_index = None

        # A dictionary mapping indexes to vocabulary
        self._index_vocab = None

        # The indexated tokens
        self._tokens_idx = None
        
        # Defining inputs placeholder for further filling
        self._X = None

        # We also need to define the labels placeholder
        self._Y = None

        # Checking if there are any tokens
        if tokens:
            # If yes, build class properties
            self._build_properties(tokens)

    @property
    def tokens(self):
        """list: A list holding tokenized words or characters.

        """

        return self._tokens

    @tokens.setter
    def tokens(self, tokens):
        self._tokens = tokens

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    @property
    def vocab_index(self):
        """dict: A dictionary mapping vocabulary to indexes.

        """

        return self._vocab_index

    @vocab_index.setter
    def vocab_index(self, vocab_index):
        self._vocab_index = vocab_index

    @property
    def index_vocab(self):
        """dict: A dictionary mapping indexes to vocabulary.

        """

        return self._index_vocab

    @index_vocab.setter
    def index_vocab(self, index_vocab):
        self._index_vocab = index_vocab

    @property
    def tokens_idx(self):
        """np.array: A numpy array holding the indexed tokens.

        """

        return self._tokens_idx

    @tokens_idx.setter
    def tokens_idx(self, tokens_idx):
        self._tokens_idx = tokens_idx

    @property
    def X(self):
        """np.array: Input samples.

        """

        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    @property
    def Y(self):
        """np.array: Target samples.

        """

        return self._Y

    @Y.setter
    def Y(self, Y):
        self._Y = Y

    def _build_properties(self, tokens):
        """Builds all properties if there are any tokens.

        Args:
            tokens (list): A list holding tokenized words or characters.

        """

        # Firstly, we need to define a tokens property
        self.tokens = tokens

        # Calculates the vocabulary and its size from tokens
        vocab = list(set(tokens))
        self.vocab_size = len(vocab)

        # Creates a dictionary mapping vocabulary to indexes
        self.vocab_index = self.vocab_to_index(vocab)

        # Creates a dictionary mapping indexes to vocabulary
        self.index_vocab = self.index_to_vocab(vocab)

        # Indexate tokens based on a vocabulary-index dictionary
        self.tokens_idx = self.indexate_tokens(tokens, self.vocab_index)

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

    def create_batches(self, X, Y, batch_size, shuffle=True):
        """Creates an interator to feed (X, Y) batches to the network.

        Args:
            X (np.array): An array of inputs.
            Y (np.array): An array of labels.
            batch_size (int): The size of each batch.
            shuffle (bool): If yes, shuffles the data.

        Yields:
            An iterator containing (X, Y) batches.

        """

        # Getting the number of avaliable samples
        n_samples = X.shape[0]

        # Calculating the number of batches
        n_batches = n_samples // batch_size

        # Creating an index vector for shuffling the data
        idx = np.arange(n_samples)

        # Checking if shuffle argument is true
        if shuffle:
            # If yes, shuffles the data
            np.random.shuffle(idx)

        # The first step should be declared as 0
        i = 0

        # Iterate through all possible batches
        for _ in range(n_batches):
            # Pre-allocate x and y batches with current batch_size
            x_batch = [None] * batch_size
            y_batch = [None] * batch_size

            # Iterate through the batch size
            for j in range(batch_size):
                # Gathers a random sample based on pre-defined index
                x_batch[j] = X[idx[i]]
                y_batch[j] = Y[idx[i]]

                # Increases to next step
                i += 1

            yield x_batch, y_batch
