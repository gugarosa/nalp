import nalp.stream.preprocess as p
import numpy as np
import tensorflow as tf
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn import RNN

sentences = "I have a hippo and a cat"
predict_sentences = "hippo and hippo hippo hippo"

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_char
)

# Applying pre-processing pipeline to X
sentences = pipe(sentences)
predict_sentences = pipe(predict_sentences)

d = OneHot(sentences, 3)

idx_token = d.indexate_tokens(predict_sentences, d.vocab_index)
x_p, y_p = d.encode_tokens(idx_token, 3, d.vocab_size)


tf.reset_default_graph()

rnn = RNN()
rnn.train(d.X, d.Y)

predict = rnn.predict(x_p)
print(predict[0])
print(d.index_vocab[predict[0][-1]])
