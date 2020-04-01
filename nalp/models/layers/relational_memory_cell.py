import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell

class RelationalMemoryCell(AbstractRNNCell):
    """
    """

    def __init__(self, n_memories=5, n_heads=2, head_size=10, n_blocks=1, n_layers=5, forget_bias=1.0, activation='tanh', **kwargs):
        """
        """

        super(RelationalMemoryCell, self).__init__(**kwargs)

        self.memory_size = n_heads * head_size
        self.units = self.memory_size * n_memories
        self.n_memories = n_memories
        self.n_heads = n_heads
        self.head_size = head_size
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.forget_bias = forget_bias
        self.activation = activation
        self.n_gates = 2


    @property
    def state_size(self):
        return self.units * 2

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.n_gates),
            name='kernel'
        )

        self.bias = self.add_weight(
            shape=(self.n_gates),
            name='bias'
        )

        self.built = True

    def call(self, inputs, states):
        return inputs, inputs