"""Audio-related corpus.
"""

from nalp.core import Corpus
from nalp.utils import loader, logging

logger = logging.get_logger(__name__)


class AudioCorpus(Corpus):
    """An AudioCorpus class is used to defined the first step of the workflow.

    It serves to load the raw audio, pre-process it and create their tokens and
    vocabulary.

    """

    def __init__(self, from_file: str, min_frequency: int = 1) -> None:
        """Initialization method.

        Args:
            from_file: An input file to load the audio.
            min_frequency: Minimum frequency of individual tokens.

        """

        logger.info("Overriding class: Corpus -> AudioCorpus.")

        super(AudioCorpus, self).__init__(min_frequency=min_frequency)

        audio = loader.load_audio(from_file)

        self.tokens = []
        for step in audio:
            if not step.is_meta and step.channel == 0 and step.type == "note_on":
                note = step.bytes()

                self.tokens.append(str(note[1]))

        self._check_token_frequency()
        self._build()

        logger.debug(
            "Tokens: %d | Type: audio | Minimum frequency: %d | Vocabulary size: %d.",
            len(self.tokens),
            self.min_frequency,
            len(self.vocab),
        )
        logger.info("AudioCorpus created.")
