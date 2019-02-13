import nalp.utils.decorators as d
import tensorflow as tf


class Neural:
    """A Neural class is responsible for holding vital information when defining a
    neural network. Note that some methods have to be redefined when using its childs.

    Properties:
        x (tensor): A placeholder of custom shape to hold input data.
        y (tensor): A placeholder of custom shape to hold input data labels.

    Methods:
        model(): This method will build the network's architecture.
        loss(): You can define your own model's loss function.
        accuracy(): Metrics are also useful when training a model, such as accuracy.
        optimizer(): Also, the optimizer that will optimize the loss function.
        predictor(): You can instanciate a custom predictor, for post-training tasks.
        predictor_prob(): You can instanciate a probability predictor, for post-training tasks.
        train(): A training function should be created.
        predict(): If neeeded, a predicting function can also be implemented.

    """

    def __init__(self, shape=None):
        """Initialization method.
        Note that basic variables shared by all childs should be declared here.

        Args:
            shape (list): A list containing in its first position the shape of the inputs (x)
            and on its second position, the shape of the labels (y).

        """

        # We need to define a placeholder for the data tensor
        self.x = tf.placeholder(
            tf.float32, shape[0], name='inputs')

        # And another for the data's labels tensor
        self.y = tf.placeholder(tf.float32, shape[1], name='labels')

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
    def accuracy(self):
        """Each child of Neural object has the possibility of defining its custom accuracy function.
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

    @d.define_scope
    def predictor_prob(self):
        """Each child of Neural object has the possibility of defining its custom predictor (probabilities).
        Please check the vanilla RNN class in order to implement your own.
        
        """

        raise NotImplementedError

    def train(self):
        """You should implement your own training step in order to work with this class.

        """

        raise NotImplementedError

    def predict(self):
        """If needed, you can implement what happens later, if you wish to restore your model and
        predict something new and return its label.

        """

        raise NotImplementedError
