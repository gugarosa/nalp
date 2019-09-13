import nalp.utils.preprocess as p
from nalp.corpus.text import TextCorpus
from nalp.encoders.onehot import OnehotEncoder
from nalp.datasets.next import NextDataset
from nalp.neurals.rnn import RNN
import tensorflow as tf

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an OnehotEncoder
encoder = OnehotEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab, corpus.vocab_size)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating next target Dataset
dataset = NextDataset(encoded_tokens, max_length=10, batch_size=128)

# Creating the RNN
rnn = RNN(vocab_size=corpus.vocab_size, hidden_size=64)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

rnn.compile(optimizer, loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# rnn.fit(dataset.batches, epochs=100)

# rnn.save_weights('out', save_format='tf')

# rnn.train_on_batch(dataset.batches.take(1))

rnn.load_weights('out')

rnn.evaluate(dataset.batches)