# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from nalp.corpus import SentenceCorpus

corpus = SentenceCorpus(
    from_file="data/sentence/coco_image_captions.txt",
    corpus_type="word",
    min_frequency=1,
    max_pad_length=10,
    sos_eos_tokens=True,
)

print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
