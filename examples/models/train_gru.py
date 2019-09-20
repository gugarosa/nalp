import tensorflow as tf

from nalp.corpus.text import TextCorpus
from nalp.datasets.next import NextDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.gru import GRU

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

# Creating the GRU
gru = GRU(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# Compiling the GRU
gru.compile(optimize=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the GRU
gru.fit(dataset.batches, epochs=100)

# Evaluating the GRU
# gru.evaluate(dataset.batches)

# Saving GRU weights
# gru.save_weights('models/gru', save_format='tf')

# Loading GRU weights
# gru.load_weights('models/gru')