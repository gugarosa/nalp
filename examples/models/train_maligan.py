# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models import MaliGAN

corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="word")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=4)

maligan = MaliGAN(
    encoder=encoder,
    vocab_size=corpus.vocab_size,
    max_length=10,
    embedding_size=256,
    hidden_size=512,
    n_filters=(64, 128, 256),
    filters_size=(3, 5, 5),
    dropout_rate=0.25,
    temperature=1,
)

maligan.compile(
    pre_optimizer=tf.optimizers.Adam(learning_rate=0.01),
    d_optimizer=tf.optimizers.Adam(learning_rate=0.001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.001),
)

maligan.pre_fit(dataset.batches, g_epochs=50, d_epochs=10)

maligan.fit(dataset.batches, epochs=10, d_epochs=5)

maligan.save_weights("trained/maligan", save_format="tf")
