import tensorflow as tf

from nalp.corpus.text import TextCorpus
from nalp.datasets.language_modeling import LanguageModelingDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.adversarial.seqgan import SeqGAN

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

# Creating the SeqGAN
seqgan = SeqGAN(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256,
                hidden_size=512, n_filters=[64, 128, 256], filters_size=[3, 5, 5], dropout_rate=0.25)

# Compiling the SeqGAN
seqgan.compile(tf.optimizers.Adam(learning_rate=0.001),
               tf.losses.SparseCategoricalCrossentropy(from_logits=True))

# Pre-fitting the SeqGAN
seqgan.pre_fit(dataset.batches, g_epochs=10, d_epochs=1, d_steps=3)
