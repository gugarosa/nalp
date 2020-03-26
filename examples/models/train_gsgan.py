import tensorflow as tf

from nalp.corpus.text import TextCorpus
from nalp.datasets.language_modeling import LanguageModelingDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.gsgan import GSGAN

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='word')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_length=10, batch_size=4)

# Creating the GSGAN
gsgan = GSGAN(encoder=encoder, vocab_size=corpus.vocab_size, max_length=10, embedding_size=256,
                hidden_size=512, n_filters=[64, 128, 256], filters_size=[3, 5, 5], dropout_rate=0.25, temperature=1)

# Compiling the GSGAN
gsgan.compile(g_optimizer=tf.optimizers.Adam(learning_rate=0.001),
               d_optimizer=tf.optimizers.Adam(learning_rate=0.001))

# As NALP's RNNs are stateful, we need to build it with a fixed batch size
gsgan.G.build((4, 10))

# Compiling the RNN
gsgan.G.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Pre-fitting the GSGAN
# gsgan.pre_fit(dataset.batches, g_epochs=50, d_epochs=10)
gsgan.G.fit(dataset.batches, epochs=100)

# # Fitting the GSGAN
# gsgan.fit(dataset.batches, epochs=10, d_epochs=5, n_rollouts=16)

# # Saving GSGAN weights
# gsgan.save_weights('trained/gsgan', save_format='tf')
