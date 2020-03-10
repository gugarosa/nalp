import tensorflow as tf

from nalp.corpus.text import TextCorpus
from nalp.datasets.language_modelling import LanguageModellingDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.recurrent.rnn import RNN

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modelling Dataset
dataset = LanguageModellingDataset(encoded_tokens, max_length=10, batch_size=64)

# Creating the RNN
rnn = RNN(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# As NALP's RNNs are stateful, we need to build it with a fixed batch size
rnn.build((64, None))

# Compiling the RNN
rnn.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the RNN
rnn.fit(dataset.batches, epochs=100)

# Evaluating the RNN
# rnn.evaluate(dataset.batches)

# Saving RNN weights
rnn.save_weights('trained/rnn', save_format='tf')
