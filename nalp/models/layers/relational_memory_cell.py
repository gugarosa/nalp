import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell, Dense, LayerNormalization
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

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = Dense(d_model)
    self.wk = Dense(d_model)
    self.wv = Dense(d_model)
    
    self.dense = Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights

class RelationalMemoryCell(AbstractRNNCell):
    """
    """

    def __init__(self, n_memories=3, n_heads=5, head_size=10, n_blocks=1, n_layers=5, forget_bias=1.0, activation='tanh', **kwargs):
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
        self.activation = math_ops.tanh
        self.n_gates = 2 * self.memory_size

        self.mlp_layers = [Dense(self.units, activation='relu') for _ in range(n_layers)]
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        self.input_projection = Dense(self.units)

        self.att = MultiHeadAttention(self.units, self.n_heads)


    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    
    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(self.units, self.n_gates),
            name='kernel'
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.n_gates),
            name='recurrent_kernel'
        )

        self.bias = self.add_weight(
            shape=(self.n_gates,),
            name='bias'
        )

        self.built = True

    def _multi_head_attention(self, inputs, memory):
        m = tf.concat([inputs, memory], axis=1)

        memory, attn = self.att(m, k=m, q=memory, mask=None)

        memory = tf.squeeze(memory, 1)

        

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

        h_prev, m_prev = states

        inputs = self.input_projection(inputs)

        inputs_f = inputs
        inputs_i = inputs

        # print(inputs_f, inputs_i)

        k_f, k_i = array_ops.split(self.kernel, num_or_size_splits=2, axis=1)
        rk_f, rk_i = array_ops.split(self.recurrent_kernel, num_or_size_splits=2, axis=1)

        print(k_f, k_i)
        print(rk_f, rk_i)

        x_f = K.dot(inputs_f, k_f)
        x_i = K.dot(inputs_i, k_i)

        x_f += K.dot(h_prev, rk_f)
        x_i += K.dot(h_prev, rk_i)


        b_f, b_i = array_ops.split(self.bias, num_or_size_splits=2, axis=0)

        x_f = K.bias_add(x_f, b_f)
        x_i = K.bias_add(x_i, b_i)


        m = self._attend_over_memory(inputs, m_prev)

        m = math_ops.sigmoid(x_f + self.forget_bias) * m_prev + math_ops.sigmoid(x_i) * math_ops.tanh(m)

        h = math_ops.tanh(m)

        return h, [h, m]