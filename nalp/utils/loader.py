"""Data-loading utilities.
"""

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

    logger.debug('Loading %s ...', file_name)

    try:
        file = open(file_name, 'rb')

        text = file.read().decode(encoding='utf-8')

        return text

    except FileNotFoundError:
        e = f'File not found: {file_name}.'

        logger.error(e)

        raise


def load_audio(file_name):
    """Loads an audio .mid file.

    Args:
        file_name (str): The file name to be loaded.

    Returns:
        A list with the loaded notes.

    """

    logger.debug('Loading %s ...', file_name)

    try:
        audio = MidiFile(file_name)

        return audio

    except FileNotFoundError:
        e = f'File not found: {file_name}.'

        logger.error(e)

        raise
