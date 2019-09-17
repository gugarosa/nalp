import tensorflow as tf
from tensorflow.keras.layers import Layer


class Linear(Layer):
    """A linear layer is an easy-abstraction to show the possibilities of implementing
    your own layer.

    """

    def __init__(self, name='linear', hidden_size=32):
        """Initialization method.

        Args:
            hidden_size (int): The amount of hidden neurons in the layer.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Linear, self).__init__(name=name)

        # A property to hold the size of the layer
        self.hidden_size = hidden_size

    def build(self, input_shape):
        """Builds the layer.

        Args:
            input_shape (tuple): A tuple holding the input's shape.

        """

        self.w = self.add_weight(
            shape=(input_shape[-1], self.hidden_size),
            initializer='glorot_uniform',
            trainable=True)

        self.b = self.add_weight(
            shape=(self.hidden_size,),
            initializer='zeros',
            trainable=True)

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The layer output after performing its calculations. This layer will
            produce y = xW + b.

        """

        # Calculating the layer (y = xW + b)
        y = tf.matmul(x, self.w) + self.b

        return y
