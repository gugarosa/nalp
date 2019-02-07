import nalp.stream.preprocess as p
import numpy as np
import tensorflow as tf
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn import RNN

sentences = "i like dog and a cat"

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_char
)

# Applying pre-processing pipeline to X
sentences = pipe(sentences)

d = OneHot(sentences, 3)


tf.reset_default_graph()

rnn = RNN()
rnn.train(d.X, d.Y)

print(d.X[0:1])
print(d.Y[0:1])

rnn.predict(d.X[0:1])
