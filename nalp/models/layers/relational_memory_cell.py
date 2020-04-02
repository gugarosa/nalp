import tensorflow as tf
from tensorflow.keras.layers import (AbstractRNNCell, Dense,
                                     LayerNormalization, dot)
from tensorflow.python.framework import constant_op, dtypes
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn_ops
from tensorflow.python.platform import tf_logging as logging

from nalp.models.layers.multi_head_attention import MultiHeadAttention


class RelationalMemoryCell(AbstractRNNCell):
    """A RelationalMemoryCell class is the one in charge of a Relational Memory cell implementation.

    References:
        A. Santoro, et al. Relational recurrent neural networks.
        Advances in neural information processing systems (2018).

    """

    def __init__(self, n_slots, n_heads, head_size, n_blocks=1, n_layers=3,
                 activation='tanh', recurrent_activation='hard_sigmoid', forget_bias=1.0,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,**kwargs):
        """Initialization method.

        Args:

        """

        # Overrides its parent class with any custom arguments if needed
        super(RelationalMemoryCell, self).__init__(**kwargs)
        
        # Number of memory slots and their sizes
        self.n_slots = n_slots
        self.slot_size = n_heads * head_size

        # Number of attention heads and their sizes
        self.n_heads = n_heads
        self.head_size = head_size

        # Number of feed-forward network blocks and their sizes
        self.n_blocks = n_blocks
        self.n_layers = n_layers

        # Activation functions
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

        # Forget gate bias value
        self.forget_bias = forget_bias

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Number of outputted units
        self.units = self.slot_size * n_slots

        # Number of outputted units from the gates
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
        """Builds up the cell according to its input shape.

        Args:
            input_shape (tf.Tensor): Tensor holding the input shape.
            
        """
        
        # Defining a property to hold the `W` matrices
        self.kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # Defining a property to hold the `U` matrices
        self.recurrent_kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        # Defining a property to hold the `b` vectors
        self.bias = self.add_weight(
            shape=(self.n_gates,),
            name='bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
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
        """Method that holds vital information whenever this class is called.

        Args:
            inputs (tf.Tensor): An input tensor.
            states (list): A list holding previous states and memories.

        Returns:
            Output states as well as current state and memory.

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
