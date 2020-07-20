import tensorflow as tf

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models.generators import RMCGenerator

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_length=10, batch_size=64, shuffle=True)

# Creating the RMC
rmc = RMCGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256,
                   n_slots=5, n_heads=5, head_size=25, n_blocks=1, n_layers=3)

# As NALP's RMCs are stateful, we need to build it with a fixed batch size
rmc.build((64, None))

# Compiling the RMC
rmc.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the RMC
rmc.fit(dataset.batches, epochs=200)

# Evaluating the RMC
# rmc.evaluate(dataset.batches)

# Saving RMC weights
rmc.save_weights('trained/rmc', save_format='tf')
