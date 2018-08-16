import pandas as pd

import feelyng.utils.logging as l
from sklearn.model_selection import train_test_split


logger = l.get_logger(__name__)


def split_data(X, Y, split_size=0.5, random_state=42):
    """Splits X, Y (samples, labels) data into training and testing sets.

    Args:
        X (obj): Input samples object.
        Y (obj): Input labels object.
        split_size (float): The proportion of test sets.
        random_state (int): Random integer to provide a random state to splitter.

    Returns:
        X and Y training, testing sets.

    """

    try:
        Y = pd.get_dummies(Y).values
    except Exception as e:
        logger.warn(e)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=split_size, random_state=42)

    return X_train, X_test, Y_train, Y_test
