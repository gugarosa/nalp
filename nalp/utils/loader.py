# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Data-loading utilities."""

from pathlib import Path

from mido import MidiFile


def load_txt(file_name: str | Path) -> str:
    """Load UTF-8 text without translating line endings.

    Args:
        file_name: Source text file path.

    Returns:
        Decoded file contents with their original line endings.

    Raises:
        OSError: The file cannot be read.
        UnicodeDecodeError: The file contents are not valid UTF-8.

    """

    return Path(file_name).read_bytes().decode("utf-8")


def load_audio(file_name: str | Path) -> MidiFile:
    """Load a MIDI file into an in-memory event representation.

    Args:
        file_name: Source MIDI file path.

    Returns:
        The parsed MIDI file.

    Raises:
        OSError: The file cannot be opened or parsed as MIDI.
        EOFError: The file ends before a complete MIDI structure is read.

    """

    return MidiFile(file_name)
