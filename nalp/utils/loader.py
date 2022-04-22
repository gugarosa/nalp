"""Data-loading utilities.
"""

from typing import List

from mido import MidiFile

from nalp.utils import logging

logger = logging.get_logger(__name__)


def load_txt(file_name: str) -> str:
    """Loads a .txt file.

    Args:
        file_name: The file name to be loaded.

    Returns:
        (str): Loaded text.

    """

    logger.debug("Loading %s ...", file_name)

    try:
        file = open(file_name, "rb")

        text = file.read().decode(encoding="utf-8")

        return text

    except FileNotFoundError:
        e = f"File not found: {file_name}."

        logger.error(e)

        raise


def load_audio(file_name: str) -> List[str]:
    """Loads an audio .mid file.

    Args:
        file_name: The file name to be loaded.

    Returns:
        (List[str]): Loaded notes.

    """

    logger.debug("Loading %s ...", file_name)

    try:
        audio = MidiFile(file_name)

        return audio

    except FileNotFoundError:
        e = f"File not found: {file_name}."

        logger.error(e)

        raise
