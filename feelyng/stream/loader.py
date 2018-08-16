import sys

import feelyng.utils.logging as l
import pandas as pd

logger = l.get_logger(__name__)


def load_csv(csv_path):
    """Loads a CSV file into a dataframe object.

    Args:
        csv_path (str): a string holding the .csv's path

    Returns:
        A Panda's dataframe object.

    """

    try:
        # Tries to read .csv file into a dataframe
        csv = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        # If file is not found, handle the exception and exit
        logger.error('Failed to open file ' + csv_path)
        raise Exception(e)

    return csv
