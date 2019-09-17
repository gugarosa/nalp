from tensorflow.keras import Model

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class SimpleNeural(Model):
    """A SimpleNeural class is responsible for easily-implementing a neural network, when
    custom training or additional sets are not needed.

    """

    def __init__(self, name):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(SimpleNeural, self).__init__(name=name)

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
