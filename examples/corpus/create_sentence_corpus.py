from nalp.corpus import SentenceCorpus

# Creating a char SentenceCorpus from file
# corpus = SentenceCorpus(from_file='data/sentence/coco_image_captions.txt', corpus_type='char',
#                         min_frequency=1, max_pad_length=10, sos_eos_tokens=True)

# Creating a word SentenceCorpus from file
corpus = SentenceCorpus(
    from_file="data/sentence/coco_image_captions.txt",
    corpus_type="word",
    min_frequency=1,
    max_pad_length=10,
    sos_eos_tokens=True,
)

# Accessing SentenceCorpus properties
print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
