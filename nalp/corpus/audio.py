# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Audio-related corpus."""

from pathlib import Path

from nalp.core.corpus import Corpus
from nalp.utils import loader


class AudioCorpus(Corpus):
    """Build a vocabulary of note-on pitches from the first MIDI channel."""

    def __init__(self, from_file: str | Path, min_frequency: int = 1) -> None:
        """Load MIDI note events and build their token mappings.

        Only channel-zero note_on messages become tokens, represented as decimal pitch strings.

        Args:
            from_file: MIDI source path.
            min_frequency: Minimum pitch-token count before replacement with the unknown-token marker.

        Raises:
            OSError: The source file cannot be read.

        """

        super().__init__(min_frequency=min_frequency)

        audio = loader.load_audio(from_file)

        self.tokens = [str(step.note) for step in audio if step.type == "note_on" and step.channel == 0]

        self._check_token_frequency()
        self._build()
