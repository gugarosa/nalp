import feelyng.encoder.count as count
import feelyng.encoder.tfidf as tfidf
import feelyng.encoder.word2vec as word2vec
import feelyng.utils.logging as l

logger = l.get_logger(__name__)


class Encoder:
    """
    """

    def __init__(self, type='count'):
        """

        Args:

        """

        logger.info('Initializing Encoder ...')

        # One should also declare the type of the encoder
        self.type = type

        # The encoder object will be initialized as None
        self.encoder = None

        # We initially set the encoded data as None
        self.encoded_data = None

        # We will log some important information
        logger.info('Encoder created.')
        logger.info('Encoder type: ' + self.type)

    def learn(self, data_to_learn):
        """
        """

        logger.debug('Running method: learn()')

        # We need to check the encoder type prior to its learning process
        if self.type == 'count':
            # Count Vectorizer
            self.encoder = count.learn_count(data_to_learn)

        elif self.type == 'tfidf':
            # TFIDF Vectorizer
            self.encoder = tfidf.learn_tfidf(data_to_learn)

        elif self.type == 'word2vec':
            # Word2Vec
            self.encoder = word2vec.learn_word2vec(data_to_learn)

    def encode(self, data_to_encode):
        """
        """

        # Check if there is an encoder that actually exists
        if not self.encoder:
            e = 'You need to call learn() prior to encode() method.'
            logger.error(e)
            raise RuntimeError(e)

        logger.debug('Running method: encode()')

        # We need to check the encoder type prior to its encoding process
        if self.type == 'count':
            # Count Vectorizer
            self.encoded_data = count.encode_count(
                self.encoder, data_to_encode)

        elif self.type == 'tfidf':
            # TFIDF Vectorizer
            self.encoded_data = tfidf.encode_tfidf(
                self.encoder, data_to_encode)

        elif self.type == 'word2vec':
            # Word2Vec
            self.encoded_data = word2vec.encode_word2vec(
                self.encoder, data_to_encode)
