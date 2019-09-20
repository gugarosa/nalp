import tensorflow as tf

import nalp.utils.preprocess as p
from nalp.corpus.text import TextCorpus
from nalp.datasets.next import NextDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.bi_lstm import BiLSTM

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating next target Dataset
dataset = NextDataset(encoded_tokens, max_length=10, batch_size=64)

# Creating the LSTM
lstm = BiLSTM(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# Compiling the LSTM
lstm.compile(optimize=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the LSTM
lstm.fit(dataset.batches, epochs=100)

# Evaluating the LSTM
# lstm.evaluate(dataset.batches)

# Saving LSTM weights
# lstm.save_weights('models/lstm', save_format='tf')

# Loading LSTM weights
# lstm.load_weights('models/lstm')