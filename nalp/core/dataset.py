import numpy as np
import tensorflow as tf


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
        """Creates an iterable to feed (X, Y) batches to the network.
        
        Args:
            X (np.array): An array of inputs.
            Y (np.array): An array of labels.
            batch_size (int): The size of each batch.
            shuffle (bool): If data should be shuffled or not.

        Returns:
            A tensorflow dataset iterable.

         """

        # Slicing dataset
        data = tf.data.Dataset.from_tensor_slices((X, Y))

        # Checking if data should be shuffled
        if shuffle:
            data = data.shuffle(len(Y))

        # Applying batches
        data = data.batch(batch_size)

        return data

    def decode(self, encoded_data):
        """Decodes array of probabilites into raw text.

        Args:
            encoded_data (np.array | tf.Tensor): An array holding probabilities.

        Returns:
            A decoded list (can be characters or words).

        """

        # Declaring a null string to hold the decoded data
        decoded_text = []

        # Iterating through all encoded data
        for e in encoded_data:
                # If probability is true, we need to recover the argmax of 'e'
                decoded_text.append(self.index_vocab[np.argmax(e)])

        return decoded_text
