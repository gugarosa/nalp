"""Data-loading utilities."""

from pathlib import Path

from mido import MidiFile


def load_txt(file_name: str | Path) -> str:
    """Load a UTF-8 text file."""

    return Path(file_name).read_bytes().decode("utf-8")


def load_audio(file_name: str) -> MidiFile:
    """Loads an audio .mid file.

    Args:
        file_name: The file name to be loaded.

    Returns:
        (MidiFile): Loaded MIDI file.

    """

    return MidiFile(file_name)
