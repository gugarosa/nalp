import nalp.stream.preprocess as p
import nalp.stream.loader as l
import numpy as np
import tensorflow as tf
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn import RNN

sentences = l.load_txt('data/chapter1_harry.txt')
pred_input = ""

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_char
)

# Applying pre-processing pipeline to X
sentences = pipe(sentences)
pred_input = pipe(pred_input)

d = OneHot(sentences, max_length=30)

tf.reset_default_graph()

rnn = RNN(max_length=d.max_length, hidden_size=128, vocab_size=d.vocab_size, learning_rate=0.01)

rnn.train(d.X, d.Y, epochs=100, verbose=True, save_model=True)

pred = rnn.predict(d.X, probability=False)
print(pred)

text = ''
for p in pred:
    for t in p:
        text += d.index_vocab[t]
print(text)

# gen_text = rnn.generate_text(dataset=d, start_text=pred_input, max_length=10)
# print(gen_text) 
