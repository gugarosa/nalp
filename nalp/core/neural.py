import tensorflow as tf

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class Neural(tf.keras.Model):
    """A Neural class is responsible for holding vital information when defining a
    neural network.

    Note that some methods have to be redefined when using its childs.

    """

    def __init__(self):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Neural, self).__init__()

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError
