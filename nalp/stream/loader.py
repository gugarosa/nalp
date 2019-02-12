import sys

import nalp.utils.logging as l
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

def load_txt(txt_path):
    """Loads a TXT file into a string.

    Args:
        txt_path (str): a string holding the .txt's path

    Returns:
        A string containing the loaded text.
        
    """

    try:
        # Tries to read .txt file
        txt_file = open(txt_path)
        # If possible, read its content
        content = txt_file.read()
    except FileNotFoundError as e:
        # If file is not found, handle the exception and exit
        logger.error('Failed to open file ' + txt_path)
        raise Exception(e)

    return content