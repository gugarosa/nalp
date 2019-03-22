import sys

import nalp.utils.logging as l
import pandas as pd

logger = l.get_logger(__name__)


def load_csv(csv_path):
    """Loads a CSV file into a dataframe object.

    Args:
        csv_path (str): A string holding the .csv's path.

    Returns:
        A Panda's dataframe object.

    """

    # Tries to read .csv file into a dataframe
    try:
        # Actually reads the .csv file
        csv = pd.read_csv(csv_path)

    # If file is not found, handle the exception and exit
    except FileNotFoundError as e:
        logger.error('Failed to open file ' + csv_path)

        raise Exception(e)

    return csv


def load_txt(txt_path):
    """Loads a TXT file into a string.

    Args:
        txt_path (str): A string holding the .txt's path.

    Returns:
        A string containing the loaded text.

    """

    # Tries to read .txt file
    try:
        # Opens the .txt file
        txt_file = open(txt_path)

        # If possible, read its content
        txt = txt_file.read()

    # If file is not found, handle the exception and exit
    except FileNotFoundError as e:
        logger.error('Failed to open file ' + txt_path)

        raise Exception(e)

    return txt
