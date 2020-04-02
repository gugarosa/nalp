import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell, Dense, LayerNormalization, dot
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


from nalp.models.layers.multi_head_attention import MultiHeadAttention

class RelationalMemoryCell(AbstractRNNCell):
    """
    """

    def __init__(self, n_slots=5, n_heads=8, head_size=20, n_blocks=1, n_layers=5, forget_bias=1.0, activation='tanh', **kwargs):
        """
        """

        super(RelationalMemoryCell, self).__init__(**kwargs)

        self.n_slots = n_slots
        self.slot_size = n_heads * head_size

        self.n_heads = n_heads
        self.head_size = head_size

        self.n_blocks = n_blocks
        self.n_layers = n_layers

        self.forget_bias = forget_bias
        self.activation = math_ops.tanh

        self.units = self.slot_size * n_slots
        self.n_gates = 2 * self.slot_size


        self.mlp_layers = [Dense(self.slot_size, activation='relu') for _ in range(n_layers)]
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        self.input_projection = Dense(self.slot_size)

        self.att = MultiHeadAttention(self.slot_size, self.n_heads)


    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    
    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name='kernel'
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name='recurrent_kernel'
        )

        self.bias = self.add_weight(
            shape=(self.n_gates,),
            name='bias'
        )

        self.built = True

    def _multi_head_attention(self, inputs, memory):
        m = tf.concat([inputs, memory], axis=1)

        memory, _ = self.att(memory, m, m)

        # memory = tf.squeeze(memory, 1)

        

        return memory

    def _attend_over_memory(self, inputs, memory):
        for _ in range(self.n_blocks):
            att_memory = self._multi_head_attention(inputs, memory)
            memory = self.norm1(att_memory + memory)
            mlp_memory = memory
            for mlp in self.mlp_layers:
                mlp_memory = mlp(mlp_memory)
            memory = self.norm2(memory + mlp_memory)

        return memory

    def call(self, inputs, states):
        """

        """

        # Gathering previous hidden and memory states
        h_prev, m_prev = states

        # print(inputs)

        inputs = tf.expand_dims(self.input_projection(inputs), 1)

        h_prev = tf.reshape(h_prev, [h_prev.shape[0], self.n_slots, self.slot_size])
        m_prev = tf.reshape(m_prev, [m_prev.shape[0], self.n_slots, self.slot_size])

        # print(inputs, h_prev)

        inputs_f = inputs
        inputs_i = inputs

        # print(inputs_f, inputs_i)

        k_f, k_i = array_ops.split(self.kernel, num_or_size_splits=2, axis=1)
        rk_f, rk_i = array_ops.split(self.recurrent_kernel, num_or_size_splits=2, axis=1)

        # print(k_f, k_i)
        # print(rk_f, rk_i)

        x_f = tf.tensordot(inputs_f, k_f, axes=[[-1], [0]])
        x_i = tf.tensordot(inputs_i, k_i, axes=[[-1], [0]])

        # x_f = tf.expand_dims(x_f, 1)
        # x_i = tf.expand_dims(x_i, 1)

        # print(x_f, x_i)

        x_f += tf.tensordot(h_prev, rk_f, axes=[[-1], [0]])
        x_i += tf.tensordot(h_prev, rk_i, axes=[[-1], [0]])

        # print(x_f, x_i)

        


        b_f, b_i = array_ops.split(self.bias, num_or_size_splits=2, axis=0)

        # print(b_f, b_i)

        x_f = K.bias_add(x_f, b_f)
        x_i = K.bias_add(x_i, b_i)

        # print(x_f, x_i)


        m = self._attend_over_memory(inputs, m_prev)

        # print(m)

        m = math_ops.sigmoid(x_f + self.forget_bias) * m_prev + math_ops.sigmoid(x_i) * math_ops.tanh(m)

        h = math_ops.tanh(m)

        m = tf.reshape(m, [m.shape[0], self.units])
        h = tf.reshape(h, [h.shape[0], self.units])

        return h, [h, m]