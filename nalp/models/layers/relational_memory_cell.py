import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell, Dense
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, LSTMStateTuple

class RelationalMemoryCell(AbstractRNNCell):
    """
    """

    def __init__(self, n_memories=5, n_heads=4, head_size=25, n_blocks=1, n_layers=5, forget_bias=1.0, activation='tanh', **kwargs):
        """
        """

        super(RelationalMemoryCell, self).__init__(**kwargs)

        self.memory_size = n_heads * head_size
        # self.units = self.memory_size * n_memories
        self.n_memories = n_memories
        self.n_heads = n_heads
        self.head_size = head_size
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.forget_bias = forget_bias
        self.activation = math_ops.tanh
        self.n_gates = 2 * self.memory_size

        self.input_projection = Dense(self.memory_size)


    @property
    def state_size(self):
        return [self.memory_size, self.memory_size]

    @property
    def output_size(self):
        return self.n_memories * self.memory_size

    
    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(self.memory_size, self.n_gates),
            name='kernel'
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.memory_size, self.n_gates),
            name='recurrent_kernel'
        )

        self.bias = self.add_weight(
            shape=(self.n_gates,),
            name='bias'
        )

        self.built = True

    def _attend_over_memory(self, inputs, memory):
        return memory

    def call(self, inputs, states):
        """

        """

        h_prev, m_prev = states
        inputs = self.input_projection(inputs)

        inputs_i = inputs
        inputs_f = inputs

        k_i, k_f = array_ops.split(self.kernel, num_or_size_splits=2, axis=1)
        rk_i, rk_f = array_ops.split(self.recurrent_kernel, num_or_size_splits=2, axis=1)

        x_i = K.dot(inputs_i, k_i)
        x_f = K.dot(inputs_f, k_f)

        x_i += K.dot(h_prev, rk_i)
        x_f += K.dot(h_prev, rk_f)


        b_i, b_f = array_ops.split(self.bias, num_or_size_splits=2, axis=0)

        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)

        m = self._attend_over_memory(inputs, m_prev)

        h = math_ops.tanh(m)

        m = math_ops.sigmoid(x_f + self.forget_bias) * m_prev + math_ops.sigmoid(x_i) * math_ops.tanh(m)


        return h, [h, m]