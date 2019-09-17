import tensorflow as tf

import nalp.utils.preprocess as p
from nalp.corpus.text import TextCorpus
from nalp.datasets.next import NextDataset
from nalp.encoders.onehot import OnehotEncoder
from nalp.models.rnn import RNN
from nalp.utils import math

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an OnehotEncoder
encoder = OnehotEncoder()

# Learns the encoding based on the TextCorpus dictionary, reverse dictionary and vocabulary size
encoder.learn(corpus.vocab_index, corpus.index_vocab, corpus.vocab_size)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating next target Dataset
dataset = NextDataset(encoded_tokens, max_length=10, batch_size=64)

# import numpy as np

# a = []
# b = []
# for x, y in dataset.batches.take(2):
#     for i in x.numpy():
#         a.append(corpus.index_vocab[np.where(i == 1)[0][0]])
#     b.append(corpus.index_vocab[np.where(y.numpy() == 1)[0][0]])
# print(''.join(a))
# print(''.join(b))

# Creating the RNN
rnn = RNN(vocab_size=corpus.vocab_size, hidden_size=512)

# Compiling the RNN
rnn.compile(optimize=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Fitting the RNN
rnn.fit(dataset.batches, epochs=100)

# Evaluating the RNN
# rnn.evaluate(dataset.batches)

# Saving RNN weights
# rnn.save_weights('models/rnn', save_format='tf')

# Loading RNN weights
# rnn.load_weights('models/rnn')

e = encoder.encode(['M', 'r', '.'])
e = tf.expand_dims(e, 0)

text_generated = []

for i in range(1000):
    # print(e)
    preds = rnn(e)
    preds = tf.squeeze(preds, 0)
    preds = preds / 0.5
    predicted_id = tf.random.categorical(preds, num_samples=1)[-1,0].numpy()
    e = tf.expand_dims(encoder.encode(corpus.index_vocab[predicted_id]), 0)
    text_generated.append(corpus.index_vocab[predicted_id])

print(''.join(text_generated))

