# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from nalp.corpus import TextCorpus
from nalp.encoders import IntegerEncoder
from nalp.models.generators import RNNGenerator

# Match the training data, model class, and parameters when restoring weights
corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="word")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)

rnn = RNNGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

rnn.load_weights("trained/rnn").expect_partial()

# Inference uses a single sequence per batch
rnn.build((1, None))
start_string = "Mr. and Mrs. Dursley"
text = rnn.generate_temperature_sampling(start=start_string.split(" "), max_length=1000, temperature=0.5)

print(start_string + " " + " ".join(text))
