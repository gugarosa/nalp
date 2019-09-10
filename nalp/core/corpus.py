import nalp.utils.loader as l
import nalp.utils.logging as log
import nalp.utils.preprocess as p

logger = log.get_logger(__name__)


class Corpus():
    """
    """

    def __init__(self, tokens=None, from_file=None, type='char'):
        """Initialization method.
        
        """

        logger.info('Creating Corpus.')

        # Checks if there are not pre-loaded tokens
        if not tokens:
            # Loads the text from file
            text = l.load_txt(from_file)

            # Creates a pipeline based on desired type
            pipe = self._create_pipeline(type)

            # Retrieve the tokens
            self.tokens = pipe(text)

        # If there are tokens
        else:
            # Gathers them to the property
            self.tokens = tokens

        # Builds the vocabulary based on the tokens
        self._build_vocabulary(self.tokens)

        logger.info('Corpus created.')

    @property
    def tokens(self):
        """list: A list of tokens.

        """

        return self._tokens

    @tokens.setter
    def tokens(self, tokens):
        self._tokens = tokens

    def _create_pipeline(self, type):
        """
        """

        #
        if type not in ['char', 'word']:
            #
            error = f'Type argument should be `char` or `word`.'

            #
            logger.error(error)

            #
            raise EnvironmentError(error)
        #
        if type == 'char':
            return p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_char)

        #
        elif type == 'word':
            return p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_word)

    def _build_vocabulary(self, tokens):
        """
        """

        # Creates the vocabulary
        self.vocab = list(set(tokens))

        # Also, gathers the vocabulary size
        self.vocab_size = len(self.vocab)

        # Creates a property mapping vocabulary to indexes
        self.vocab_index = {c: i for i, c in enumerate(self.vocab)}

        # Creates a property mapping indexes to vocabulary
        self.index_vocab = {i: c for i, c in enumerate(self.vocab)}
