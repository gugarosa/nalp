import nalp.utils.decorators as d
import tensorflow as tf


class Neural:
    """A Neural class is responsible for holding vital information when defining a
    neural network. Note that some methods have to be redefined when using its childs.

    Methods:
        model(): This method will build the network's architecture.
        loss(): You can define your own model's loss function.
        optimizer(): Also, the optimizer that will optimize the loss function.

    """

    def __init__(self, step_size=2, n_class=7):
        """Initialization method.
        Note that basic variables shared by all childs should be declared here.

        Args:
            ????

        """

        self.x = tf.placeholder(
            tf.float32, [None, step_size, n_class], name='inputs')
        self.y = tf.placeholder(tf.float32, [None, n_class], name='labels')

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
