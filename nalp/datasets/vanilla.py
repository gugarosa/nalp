import nalp.utils.logging as l
import numpy as np
from nalp.core.dataset import Dataset

logger = l.get_logger(__name__)


class Vanilla(Dataset):
    """A Vanilla dataset can be seen as a regular dataset, composed by inputs and labels (X, Y).
    Note that the inputs have to be tokenized prior to instanciating this class.

    """

    def __init__(self, tokens, labels, categorical=True):
        """Initizaliation method.

        Args:
            tokens (list): A list holding tokenized words or characters.
            labels (list): A list holding the labels for each sample.
            categorical (bool): If yes, apply categorical encoding to labels.

        """

        logger.info('Overriding class: Dataset -> Vanilla.')

        # Overrides its parent class with any custom arguments if needed
        super(Vanilla, self).__init__()

        # List of unique labels
        self._unique_labels = None

        # Number of classes, derived from list of labels
        self._n_class = None

        # A dictionary mapping labels to indexes
        self._labels_index = None

        # A dictionary mapping indexes to labels
        self._index_labels = None

        # Populating X from list of tokens
        self.X = tokens

        # Check if categorical boolean is true
        if categorical:
            # If yes, calls method to convert string or integer labels into categorical
            self.Y = self._labels_to_categorical(labels)
        else:
            # If not, just apply to property
            self.Y = labels

        # Logging some important information
        logger.debug(
            f'X: {self.X.shape} | Y: {self.Y.shape}.')

        logger.info('Class overrided.')

    @property
    def unique_labels(self):
        """list: List of unique labels.

        """

        return self._unique_labels

    @unique_labels.setter
    def unique_labels(self, unique_labels):
        self._unique_labels = unique_labels

    @property
    def n_class(self):
        """int: Number of classes, derived from list of labels.

        """

        return self._n_class

    @n_class.setter
    def n_class(self, n_class):
        self._n_class = n_class

    @property
    def labels_index(self):
        """dict: A dictionary mapping labels to indexes.

        """

        return self._labels_index

    @labels_index.setter
    def labels_index(self, labels_index):
        self._labels_index = labels_index

    @property
    def index_labels(self):
        """dict: A dictionary mapping indexes to labels.

        """

        return self._index_labels

    @index_labels.setter
    def index_labels(self, index_labels):
        self._index_labels = index_labels

    def _labels_to_categorical(self, labels):
        """Maps labels into a categorical encoding.

        Args:
            labels (list): A list holding the labels for each sample.

        Returns:
            Categorical encoding of list of labels.

        """

        # Gathering unique labels
        self.unique_labels = set(labels)

        # We also need the number of classes
        self.n_class = len(self.unique_labels)

        # Creating a dictionary to map labels to indexes
        self.labels_index = {c: i for i, c in enumerate(self.unique_labels)}

        # Creating a dictionary to map indexes to labels
        self.index_labels = {i: c for i, c in enumerate(self.unique_labels)}

        # Creating a numpy array to hold categorical labels
        categorical_labels = np.zeros(
            (len(labels), self.n_class), dtype=np.int32)

        # Iterating through all labels
        for i, l in enumerate(labels):
            # Apply to current index the categorical encoding
            categorical_labels[i] = np.eye(self.n_class)[
                self.labels_index[l]]

        return categorical_labels
