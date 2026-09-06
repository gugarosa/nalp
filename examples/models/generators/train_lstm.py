# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models.generators import LSTMGenerator

corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="char")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=64)

lstm = LSTMGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# Stateful recurrent models require a fixed training batch size
lstm.build((64, None))
lstm.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

lstm.fit(dataset.batches, epochs=100)

lstm.save_weights("trained/lstm", save_format="tf")
