from tensorflow.keras import layers
import tensorflow as tf

class Linear(layers.Layer):
  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='glorot_uniform',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='zeros',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b