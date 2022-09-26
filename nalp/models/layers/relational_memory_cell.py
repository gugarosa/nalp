"""Relational-Memory Cell layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    AbstractRNNCell,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.python.keras import activations, constraints, initializers, regularizers


class RelationalMemoryCell(AbstractRNNCell):
    """A RelationalMemoryCell class is the one in charge of a Relational Memory cell implementation.

    References:
        A. Santoro, et al. Relational recurrent neural networks.
        Advances in neural information processing systems (2018).

    """

    def __init__(
        self,
        n_slots: int,
        n_heads: int,
        head_size: int,
        n_blocks: Optional[int] = 1,
        n_layers: Optional[int] = 3,
        activation: Optional[str] = "tanh",
        recurrent_activation: Optional[str] = "hard_sigmoid",
        forget_bias: Optional[float] = 1.0,
        kernel_initializer: Optional[str] = "glorot_uniform",
        recurrent_initializer: Optional[str] = "orthogonal",
        bias_initializer: Optional[str] = "zeros",
        kernel_regularizer: Optional[str] = None,
        recurrent_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        recurrent_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        **kwargs
    ):
        """Initialization method.

        Args:
            n_slots: Number of memory slots.
            n_heads: Number of attention heads.
            head_size: Size of each attention head.
            n_blocks: Number of feed-forward networks.
            n_layers: Amout of layers per feed-forward network.
            activation: Output activation function.
            recurrent_activation: Recurrent step activation function.
            forget_bias: Forget gate bias values.
            kernel_initializer: Kernel initializer function.
            recurrent_initializer: Recurrent kernel initializer function.
            bias_initializer: Bias initializer function.
            kernel_regularizer: Kernel regularizer function.
            recurrent_regularizer: Recurrent kernel regularizer function.
            bias_regularizer: Bias regularizer function.
            kernel_constraint: Kernel constraint function.
            recurrent_constraint: Recurrent kernel constraint function.
            bias_constraint: Bias constraint function.

        """

        super(RelationalMemoryCell, self).__init__(**kwargs)

        self.n_slots = n_slots
        self.slot_size = n_heads * head_size

        self.n_heads = n_heads
        self.head_size = head_size

        self.n_blocks = n_blocks
        self.n_layers = n_layers

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

        self.forget_bias = forget_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.units = self.slot_size * n_slots
        self.n_gates = 2 * self.slot_size

        self.projector = Dense(self.slot_size)

        self.before_norm = LayerNormalization()
        self.linear = [
            Dense(self.slot_size, activation="relu") for _ in range(n_layers)
        ]
        self.after_norm = LayerNormalization()

        self.attn = MultiHeadAttention(self.slot_size, self.n_heads)

    @property
    def state_size(self) -> List[int, int]:
        return [self.units, self.units]

    @property
    def output_size(self) -> int:
        return self.units

    def build(self, input_shape: tf.Tensor) -> None:
        """Builds up the cell according to its input shape.

        Args:
            input_shape: Tensor holding the input shape.

        """

        self.kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        self.bias = self.add_weight(
            shape=(self.n_gates,),
            name="bias",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        self.built = True

    def _attend_over_memory(self, inputs: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """Performs an Attention mechanism over the current memory.

        Args:
            inputs: An input tensor.
            memory: Current memory tensor.

        Returns:
            (tf.Tensor): Updated current memory based on Multi-Head Attention mechanism.

        """

        for _ in range(self.n_blocks):
            concat_memory = tf.concat([inputs, memory], 1)

            att_memory, _ = self.attn(
                memory, concat_memory, return_attention_scores=True
            )
            norm_memory = self.before_norm(att_memory + memory)

            linear_memory = norm_memory
            for layer in self.linear:
                linear_memory = layer(linear_memory)

            memory = self.after_norm(norm_memory + linear_memory)

        return memory

    def call(
        self, inputs: tf.Tensor, states: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Method that holds vital information whenever this class is called.

        Args:
            inputs: An input tensor.
            states: A list holding previous states and memories.

        Returns:
            (Tuple[tf.Tensor, List[tf.Tensor]]): Output states as well as current state and memory.

        """

        # Gathering previous hidden and memory states
        h_prev, m_prev = states

        # Projecting the inputs to the same size as the memory
        inputs = tf.expand_dims(self.projector(inputs), 1)

        # Reshaping the previous hidden state tensor
        h_prev = tf.reshape(h_prev, [h_prev.shape[0], self.n_slots, self.slot_size])

        # Reshaping the previous memory tensor
        m_prev = tf.reshape(m_prev, [m_prev.shape[0], self.n_slots, self.slot_size])

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
            x_f + self.forget_bias
        ) * m_prev + self.recurrent_activation(x_i) * self.activation(att_m)

        # Calculating current hidden state
        h = self.activation(m)

        # Reshaping both the current hidden and memory states to their correct output size
        h = tf.reshape(h, [h.shape[0], self.units])
        m = tf.reshape(m, [m.shape[0], self.units])

        return h, [h, m]

    def get_initial_state(
        self,
        inputs: Optional[tf.Tensor] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[tf.DType] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gets the cell initial state by creating an identity matrix.

        Args:
            inputs: An input tensor.
            batch_size: Size of the batch.
            dtype: Dtype from input tensor.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor]): Initial states.

        """

        states = tf.eye(self.n_slots, batch_shape=[batch_size])

        if self.slot_size > self.n_slots:
            diff = self.slot_size - self.n_slots

            padding = tf.zeros((batch_size, self.n_slots, diff))
            states = tf.concat([states, padding], -1)
        elif self.slot_size < self.n_slots:
            states = states[:, :, : self.slot_size]

        states = tf.reshape(states, (states.shape[0], -1))

        return states, states

    def get_config(self) -> Dict[str, Any]:
        """Gets the configuration of the layer for further serialization.

        Returns:
            (Dict[str, Any]): Configuration dictionary.

        """

        config = {
            "n_slots": self.n_slots,
            "slot_size": self.slot_size,
            "n_heads": self.n_heads,
            "head_size": self.head_size,
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "forget_bias": self.forget_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "units": self.units,
            "n_gates": self.n_gates,
        }
        base_config = super(RelationalMemoryCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
