import tensorflow as tf

from nalp.corpus.text import TextCorpus
from nalp.datasets.language_modeling import LanguageModelingDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.relgan import RelGAN

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_length=10, batch_size=64)

# Creating the RelGAN
relgan = RelGAN(encoder=encoder, vocab_size=corpus.vocab_size, max_length=10,
                embedding_size=256, n_slots=5, n_heads=5, head_size=25, n_blocks=1, n_layers=3,
                n_filters=[64, 128, 256], filters_size=[3, 5, 5], dropout_rate=0.25, tau=5)

# Compiling the GSGAN
relgan.compile(pre_optimizer=tf.optimizers.Adam(learning_rate=0.01),
               g_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
               d_optimizer=tf.optimizers.Adam(learning_rate=0.0001))

# Pre-fitting the RelGAN
relgan.pre_fit(dataset.batches, epochs=200)

# Fitting the RelGAN
relgan.fit(dataset.batches, epochs=50)

# Saving RelGAN weights
relgan.save_weights('trained/relgan', save_format='tf')
