import nalp.utils.logging as l

logger = l.get_logger(__name__)


def load_txt(file_name):
    """Loads a .txt file.

    Args:
        file_name (str): The file name to be loaded.

    Returns:
        A string with the loaded text.

    """

    logger.debug(f'Loading {file_name} ...')

    # Opens the .txt file and tries to read the text
    text = open(file_name, 'rb').read().decode(encoding='utf-8')

    return text
