import nalp.stream.preprocess as p
import numpy as np
import tensorflow as tf
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn import RNN

sentences = "I have a hippo and a cat. I have a hippo and a cat. I have a hippo and a cat. I have a hippo and a cat. I have a hippo and a cat."
pred_input = "hipp"

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_char
)

# Applying pre-processing pipeline to X
sentences = pipe(sentences)
pred_input = pipe(pred_input)

d = OneHot(sentences, max_length=3)

tf.reset_default_graph()

rnn = RNN(max_length=d.max_length, hidden_size=128, vocab_size=d.vocab_size)
rnn.train(d.X, d.Y, epochs=5000, verbose=False, save_model=True)

pred_text = rnn.predict(pred_input, d, length=100) 
print(pred_text)
