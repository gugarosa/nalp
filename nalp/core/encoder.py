class Encoder:
    """An Encoder class is responsible for receiving raw data and
    enconding it on a representation (i.e., count vectorizer, tfidf, word2vec).

    Properties:
        encoder (obj): An encoder generic object depending on its type (inherit objects from 
        embedding algorithms).
        encoded_data (np.array): A numpy array holding the encoded data representation.

    Methods:
        learn(): It learns an encoding representation based on child's method.
        encode(): It encodes based on previous learning.

    """

    def __init__(self):
        """Initialization method.
        Some basic shared variables between Encoder's childs should be declared here.

        """

        # The encoder object will be initialized as None
        self.encoder = None

        # We also set the encoded data as None
        self.encoded_data = None

    def learn(self):
        """This method learns an encoding representation. Note that for each child,
        you need to define your own learning algorithm (representation).

        """

        raise NotImplementedError

    def encode(self):
        """This method encodes new data based on previous learning. Also, note that you
        need to define your own encoding algorithm when using its childs.

        """

        raise NotImplementedError
