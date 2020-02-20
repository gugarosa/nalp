import tensorflow as tf

from nalp.corpus.text import TextCorpus
from nalp.datasets.next import NextDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.lstm import LSTM

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
lstm = LSTM(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# As NALP's LSTMs are stateful, we need to build it with a fixed batch size
lstm.build((64, None))

# Compiling the LSTM
lstm.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the LSTM
lstm.fit(dataset.batches, epochs=100)

# Evaluating the LSTM
# lstm.evaluate(dataset.batches)

# Saving LSTM weights
lstm.save_weights('models/lstm', save_format='tf')
