import nalp.stream.loader as l
import nalp.stream.preprocess as p
import numpy as np
import tensorflow as tf
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn import RNN

# Loading a text
sentences = l.load_txt('data/chapter1_harry.txt')

# Defining a predition input
pred_input = "Mr. Dursley was the director of a firm called Grunnings"

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.tokenize_to_char
)

# Applying pre-processing pipeline to sentences and pred_input
sentences = pipe(sentences)
pred_input = pipe(pred_input)

# Creating a OneHot dataset
d = OneHot(sentences, max_length=10)

# Defining a neural network based on vanilla RNN
rnn = RNN(max_length=d.max_length, hidden_size=64,
          vocab_size=d.vocab_size, learning_rate=0.01)

# Training the network
rnn.train(d.X, d.Y, epochs=200, verbose=True, save_model=True)

# Predicting using the same input (just for checking what is has learnt)
pred = rnn.predict(d.X, probability=False)

# Iterating through prediction and creating a string to check predictions
pred_text = ''
for p in pred:
    for t in p:
        pred_text += d.index_vocab[t]
print(pred_text)

# Generating new text
gen_text = rnn.generate_text(dataset=d, start_text=pred_input, max_length=100)
print(gen_text)
