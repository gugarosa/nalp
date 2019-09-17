import tensorflow as tf

import nalp.utils.preprocess as p
from nalp.corpus.text import TextCorpus
from nalp.datasets.next import NextDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.embedded_rnn import EmbeddedRNN

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating next target Dataset
dataset = NextDataset(encoded_tokens, max_length=10, batch_size=128)

# Creating the EmbeddedRNN
rnn = EmbeddedRNN(vocab_size=corpus.vocab_size, embedding_size=100, hidden_size=64)

# Compiling the EmbeddedRNN
rnn.compile(optimize=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Fitting the EmbeddedRNN
rnn.fit(dataset.batches, epochs=100)

# Evaluating the EmbeddedRNN
# rnn.evaluate(dataset.batches)

# Saving EmbeddedRNN weights
# rnn.save_weights('models/embedded_rnn', save_format='tf')

# Loading EmbeddedRNN weights
# rnn.load_weights('models/embedded_rnn')
