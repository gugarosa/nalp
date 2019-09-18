from mido import MidiFile

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

    # Tries to load the file
    try:
        # Opens the .txt file
        file = open(file_name, 'rb')

        # Reads the text
        text = file.read().decode(encoding='utf-8')

        return text

    # If file can not be loaded
    except FileNotFoundError:
        # Creates an error
        e = f'File not found: {file_name}.'

        # Logs the error
        logger.error(e)

        raise


def load_doc(file_name):
    """Loads a document .txt file.

    Args:
        file_name (str): The file name to be loaded.

    Returns:
        A list with the loaded sentences.

    """

    logger.debug(f'Loading {file_name} ...')

    # Tries to load the file
    try:
        # Opens the document file
        file = open(file_name, 'rb')

        # Reads the sentences
        doc = file.read().decode(encoding='utf-8').splitlines()

        return doc

    # If file can not be loaded
    except FileNotFoundError:
        # Creates an error
        e = f'File not found: {file_name}.'

        # Logs the error
        logger.error(e)

        raise


def load_audio(file_name):
    """Loads an audio .mid file.

    Args:
        file_name (str): The file name to be loaded.

    Returns:
        A list with the loaded notes.

    """

    logger.debug(f'Loading {file_name} ...')

    # Tries to load the file
    try:
        # Opens the audio file
        audio = MidiFile(file_name)

        return audio

    # If file can not be loaded
    except FileNotFoundError:
        # Creates an error
        e = f'File not found: {file_name}.'

        # Logs the error
        logger.error(e)

        raise
