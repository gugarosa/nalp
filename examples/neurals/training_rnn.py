import nalp.stream.loader as l
import nalp.stream.preprocess as p
import numpy as np
import tensorflow as tf
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn import RNN

# Loading a text
sentences = l.load_txt('data/chapter1_harry.txt')

# Defining a predition input
pred_input = "Mr. Dursley"

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.tokenize_to_char
)

# Applying pre-processing pipeline to sentences and pred_input
sentences = pipe(sentences)
pred_input = pipe(pred_input)

# Creating a OneHot dataset
d = OneHot(sentences, max_length=10)

# Creating tensor shapes
X_SHAPE = [None, None, d.vocab_size]
Y_SHAPE = [None, d.vocab_size]

# Defining a neural network based on vanilla RNN
rnn = RNN(max_length=d.max_length, hidden_size=64,
          vocab_size=d.vocab_size, learning_rate=0.01,
          shape=[X_SHAPE, Y_SHAPE])

# Training the network
rnn.train(dataset=d, epochs=100, batch_size=128, verbose=True, save_model=True)

# Predicting using the same input (just for checking what is has learnt)
pred = rnn.predict(d.X, probability=False)

# Calling decoding function to check the predictions
# Note that if the network was predicted without probability, the decoder is also without
pred_text = d.decode(pred[0], probability=False)
print(''.join(pred_text))

# Generating new text
gen_text = rnn.generate_text(dataset=d, start_text=pred_input, length=100)
print(''.join(gen_text))
