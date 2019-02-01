import nalp.encoders.count as count
import nalp.encoders.tfidf as tfidf
import nalp.encoders.word2vec as word2vec
import nalp.utils.logging as l

logger = l.get_logger(__name__)


class Encoder:
    """An Encoder class is responsible for receiving raw data and
    enconding it on a representation (i.e., count vectorizer, tfidf, word2vec).

    Properties:
        type (str): The type of the encoder.
        encoder (obj): An encoder generic object depending on its type (inherit objects from 
        learning algorithms).
        encoded_data (np.array): A numpy array holding the encoded data representation.

    Methods:
        learn(data_to_learn): Learns an encoding representation for its parameter.
        encode(data_to_encode): Enconde its parameter based on previous learning.

    """

    def __init__(self, type='count'):
        """Initialization method.

        Args:
            type (str): The type of the encoder.

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
        """The method for learning an encoding representation. Currently, a bunch
        of 'ifs' statements.

        Args:
            data_to_learn (df): A Panda's dataframe column holding sentences to be learned.

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
        """The method for encoding new data based on previous learning. Note that,
        to invoke this class you need to call learn() first and certify thay your
        'self.enconder' object exists.

        Args:
            data_to_encode (df): A Panda's dataframe column holding sentences to be encoded.

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
