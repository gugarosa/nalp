"""Data-loading utilities.
"""

from mido import MidiFile


def load_audio(file_name: str) -> MidiFile:
    """Loads an audio .mid file.

    Args:
        file_name: The file name to be loaded.

    Returns:
        (MidiFile): Loaded MIDI file.

    """

    return MidiFile(file_name)
