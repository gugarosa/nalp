import nalp.utils.loader as l
import nalp.utils.logging as log
import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus

logger = log.get_logger(__name__)


class DocumentCorpus(Corpus):
    """A DocumentCorpus class is used to defined the first step of the workflow.

    It serves to load the document, pre-process it and create its tokens.

    """

    def __init__(self, tokens=None, from_file=None):
        """Initialization method.

        Args:
            tokens (list): A list of tokens.
            from_file (str): An input file to load the text.

        """

        logger.info('Overriding class: Corpus -> DocumentCorpus.')

        # Overrides its parent class with any custom arguments if needed
        super(DocumentCorpus, self).__init__()

        # Checks if there are not pre-loaded tokens
        if not tokens:
            # Loads the document from file
            doc = l.load_doc(from_file)

            # Creates a pipeline
            pipe = p.pipeline(p.lower_case, p.valid_char)

            # Retrieve the tokens
            self.tokens = [pipe(sent) for sent in doc]

        # If there are tokens
        else:
            # Gathers them to the property
            self.tokens = tokens

        logger.info('DocumentCorpus created.')
