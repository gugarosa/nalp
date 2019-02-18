import nalp.utils.logging as l
import numpy as np
from nalp.core.dataset import Dataset

logger = l.get_logger(__name__)


class Vanilla(Dataset):
    """A Vanilla dataset can be seen as a regular dataset, composed by inputs and labels (X, Y).
    Note that the inputs have to be tokenized prior to instanciating this class.

    Properties:
        unique_labels(list): List of unique labels.
        n_class(int): Number of classes, derived from list of labels
        labels_index(dict): A dictionary mapping labels to indexes.
        index_labels(dict): A dictionary mapping indexes to labels.

    Methods:
        _labels_to_categorical(labels): Maps labels into a categorical encoding.

    """

    def __init__(self, tokens, labels, categorical=True):
        """Initizaliation method.

        Args:
            tokens (list): A list holding tokenized words or characters.
            labels (list): A list holding the labels for each sample.
            categorical (boolean): If yes, apply categorical encoding to labels.

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
        self._X = tokens

        # Check if categorical boolean is true
        if categorical:
            # If yes, calls method to convert string or integer labels into categorical
            self._Y = self._labels_to_categorical(labels)
        else:
            # If not, just apply to property
            self._Y = labels

        # Logging some important information
        logger.debug(
            f'X: {self._X.shape} | Y: {self._Y.shape}.')

        logger.info('Class overrided.')

    @property
    def unique_labels(self):
        """List of unique labels.

        """

        return self._unique_labels

    @property
    def n_class(self):
        """Number of classes, derived from list of labels.

        """

        return self._n_class

    @property
    def labels_index(self):
        """A dictionary mapping labels to indexes.

        """

        return self._labels_index

    @property
    def index_labels(self):
        """A dictionary mapping indexes to labels.

        """

        return self._index_labels

    def _labels_to_categorical(self, labels):
        """Maps labels into a categorical encoding.

        Args:
            labels(list): A list holding the labels for each sample.

        Returns:
            Categorical encoding of list of labels.

        """

        # Gathering unique labels
        self._unique_labels = set(labels)

        # We also need the number of classes
        self._n_class = len(self._unique_labels)

        # Creating a dictionary to map labels to indexes
        self._labels_index = {c: i for i, c in enumerate(self._unique_labels)}

        # Creating a dictionary to map indexes to labels
        self._index_labels = {i: c for i, c in enumerate(self._unique_labels)}

        # Creating a numpy array to hold categorical labels
        categorical_labels = np.zeros(
            (len(labels), self._n_class), dtype=np.int32)

        # Iterating through all labels
        for i, l in enumerate(labels):
            # Apply to current index the categorical encoding
            categorical_labels[i] = np.eye(self._n_class)[
                self._labels_index[l]]

        return categorical_labels
