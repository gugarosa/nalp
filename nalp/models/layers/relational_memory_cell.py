"""Relational-Memory Cell layer.
"""

import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell, Dense, LayerNormalization, MultiHeadAttention
from tensorflow.python.keras import (activations, constraints, initializers,
                                     regularizers)


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
                 kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, **kwargs):
        """Initialization method.

        Args:
            n_slots (int): Number of memory slots.
            n_heads (int): Number of attention heads.
            head_size (int): Size of each attention head.
            n_blocks (int): Number of feed-forward networks.
            n_layers (int): Amout of layers per feed-forward network.
            activation (str): Output activation function.
            recurrent_activation (str): Recurrent step activation function.
            forget_bias (float): Forget gate bias values.
            kernel_initializer (str): Kernel initializer function.
            recurrent_initializer (str): Recurrent kernel initializer function.
            bias_initializer (str): Bias initializer function.
            kernel_regularizer (str): Kernel regularizer function.
            recurrent_regularizer (str): Recurrent kernel regularizer function.
            bias_regularizer (str): Bias regularizer function.
            kernel_constraint (str): Kernel constraint function.
            recurrent_constraint (str): Recurrent kernel constraint function.
            bias_constraint (str): Bias constraint function.

        """

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

        # `W`, `U` and `b` initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # `W`, `U` and `b` regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # `W`, `U` and `b` constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Number of outputted units
        self.units = self.slot_size * n_slots

        # Number of outputted units from the gates
        self.n_gates = 2 * self.slot_size

        # Creating a layer for projecting the input
        self.projector = Dense(self.slot_size)

        # Creating the feed-forward network
        # It is composed by linear layers and normalization ones
        self.before_norm = LayerNormalization()
        self.linear = [Dense(self.slot_size, activation='relu')
                       for _ in range(n_layers)]
        self.after_norm = LayerNormalization()

        # Creating the Multi-Head Attention layer
        self.attn = MultiHeadAttention(self.slot_size, self.n_heads)

    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        """Builds up the cell according to its input shape.

        Args:
            input_shape (tf.tensor): Tensor holding the input shape.

        """

        # Defining a property to hold the `W` matrices
        self.kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        # Defining a property to hold the `U` matrices
        self.recurrent_kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint
        )

        # Defining a property to hold the `b` vectors
        self.bias = self.add_weight(
            shape=(self.n_gates,),
            name='bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint
        )

        # Marking the built property as `True`
        self.built = True

    def _attend_over_memory(self, inputs, memory):
        """Performs an Attention mechanism over the current memory.

        Args:
            inputs (tf.tensor): An input tensor.
            memory (tf.tensor): Current memory tensor.

        Returns:
            Updated current memory based on Multi-Head Attention mechanism.

        """

        # For every feed-forward network
        for _ in range(self.n_blocks):
            # Concatenates the inputs with the memory
            concat_memory = tf.concat([inputs, memory], 1)

            # Passes down the multi-head attention layer
            att_memory, _ = self.attn(memory, concat_memory, return_attention_scores=True)

            # Passes down the first normalization layer
            norm_memory = self.before_norm(att_memory + memory)

            # Makes a copy to feed the linear layers
            linear_memory = norm_memory

            # For every linear layer
            for l in self.linear:
                # Passes down the layer
                linear_memory = l(linear_memory)

            # Calculates the final memory from the network with another normalization layer
            memory = self.after_norm(norm_memory + linear_memory)

        return memory

    def call(self, inputs, states):
        """Method that holds vital information whenever this class is called.

        Args:
            inputs (tf.tensor): An input tensor.
            states (list): A list holding previous states and memories.

        Returns:
            Output states as well as current state and memory.

        """

        # Gathering previous hidden and memory states
        h_prev, m_prev = states

        # Projecting the inputs to the same size as the memory
        inputs = tf.expand_dims(self.projector(inputs), 1)

        # Reshaping the previous hidden state tensor
        h_prev = tf.reshape(
            h_prev, [h_prev.shape[0], self.n_slots, self.slot_size])

        # Reshaping the previous memory tensor
        m_prev = tf.reshape(
            m_prev, [m_prev.shape[0], self.n_slots, self.slot_size])

        # Copying the inputs for the forget and input gates
        inputs_f = inputs
        inputs_i = inputs

        # Splitting up the kernel into forget and input gates kernels
        k_f, k_i = tf.split(self.kernel, 2, axis=1)

        # Calculating the forget and input gates kernel outputs
        x_f = tf.tensordot(inputs_f, k_f, axes=[[-1], [0]])
        x_i = tf.tensordot(inputs_i, k_i, axes=[[-1], [0]])

        # Splitting up the recurrent kernel into forget and input gates kernels
        rk_f, rk_i = tf.split(self.recurrent_kernel, 2, axis=1)

        # Calculating the forget and input gates recurrent kernel outputs
        x_f += tf.tensordot(h_prev, rk_f, axes=[[-1], [0]])
        x_i += tf.tensordot(h_prev, rk_i, axes=[[-1], [0]])

        # Splitting up the bias into forget and input gates biases
        b_f, b_i = tf.split(self.bias, 2, axis=0)

        # Adding the forget and input gate bias
        x_f = tf.nn.bias_add(x_f, b_f)
        x_i = tf.nn.bias_add(x_i, b_i)

        # Calculating the attention mechanism over the previous memory
        att_m = self._attend_over_memory(inputs, m_prev)

        # Calculating current memory state
        m = self.recurrent_activation(
            x_f + self.forget_bias) * m_prev + self.recurrent_activation(x_i) * self.activation(att_m)

        # Calculating current hidden state
        h = self.activation(m)

        # Reshaping both the current hidden and memory states to their correct output size
        h = tf.reshape(h, [h.shape[0], self.units])
        m = tf.reshape(m, [m.shape[0], self.units])

        return h, [h, m]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Gets the cell initial state by creating an identity matrix.

        Args:
            inputs (tf.tensor): An input tensor.
            batch_size (int): Size of the batch.
            dtype (dtype): Dtype from input tensor.

        Returns:
            Initial states.

        """

        # Creates an identity matrix
        states = tf.eye(self.n_slots, batch_shape=[batch_size])

        # If the slot size is bigger than number of slots
        if self.slot_size > self.n_slots:
            # Calculates its difference
            diff = self.slot_size - self.n_slots

            # Creates a new tensor for padding
            padding = tf.zeros((batch_size, self.n_slots, diff))

            # Concatenates the initial states with the padding
            states = tf.concat([states, padding], -1)

        # If the slot size is smaller than number of slots
        elif self.slot_size < self.n_slots:
            # Just gather the tensor until the desired size
            states = states[:, :, :self.slot_size]

        # Reshapes to a flatten output
        states = tf.reshape(states, (states.shape[0], -1))

        return states, states

    def get_config(self):
        """Gets the configuration of the layer for further serialization.

        """

        config = {
            'n_slots': self.n_slots,
            'slot_size': self.slot_size,
            'n_heads': self.n_heads,
            'head_size': self.head_size,
            'n_blocks': self.n_blocks,
            'n_layers': self.n_layers,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'forget_bias': self.forget_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'units': self.units,
            'n_gates': self.n_gates
        }
        base_config = super(RelationalMemoryCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
