import nalp.utils.decorators as d
import tensorflow as tf


class Neural:
    """A Neural class is responsible for holding vital information when defining a
    neural network. Note that some methods have to be redefined when using its childs.

    Methods:
        model(): This method will build the network's architecture.
        loss(): You can define your own model's loss function.
        optimizer(): Also, the optimizer that will optimize the loss function.
        predictor(): You can instanciate a custom predictor, for post-training tasks.

    """

    def __init__(self, max_length=1, vocab_size=1):
        """Initialization method.
        Note that basic variables shared by all childs should be declared here.

        Args:
            max_length (int): The maximum length of the encoding.
            vocab_size (int): The size of the vocabulary should equal the number
            of classes.

        """

        # We need to define a placeholder for the data tensor
        self.x = tf.placeholder(
            tf.float32, [None, max_length, vocab_size], name='inputs')

        # And another for the data's labels tensor
        self.y = tf.placeholder(tf.float32, [None, vocab_size], name='labels')

    @d.define_scope
    def model(self):
        """Each child of Neural object has the possibility of defining its own architecture.
        Please check the vanilla RNN class in order to implement your own.

        """

        raise NotImplementedError

    @d.define_scope
    def loss(self):
        """Each child of Neural object has the possibility of defining its custom loss function.
        Please check the vanilla RNN class in order to implement your own.

        """

        raise NotImplementedError

    @d.define_scope
    def optimizer(self):
        """Each child of Neural object has the possibility of defining its custom optimizer.
        Please check the vanilla RNN class in order to implement your own.
        
        """

        raise NotImplementedError

    @d.define_scope
    def predictor(self):
        """Each child of Neural object has the possibility of defining its custom predictor.
        Please check the vanilla RNN class in order to implement your own.
        
        """

        raise NotImplementedError

    def train(self):
        """You should implement your own training step in order to work with this class.
        
        """

        raise NotImplementedError

    def predict(self):
        """If needed, you can implement what happens later, if you wish to restore your model and
        predict something new.

        """

        raise NotImplementedError
