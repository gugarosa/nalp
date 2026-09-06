# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from nalp.corpus import TextCorpus
from nalp.encoders import IntegerEncoder
from nalp.models import SeqGAN

# Match the training data, model class, and parameters when restoring weights
corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="char")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)

seqgan = SeqGAN(
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

seqgan.load_weights("trained/seqgan").expect_partial()

# Inference uses a single sequence per batch
seqgan.G.build((1, None))
start_string = "Mr."
text = seqgan.G.generate_temperature_sampling(start=start_string, max_length=1000, temperature=1)

print(start_string + "".join(text))
