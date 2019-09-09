import nalp.utils.preprocess as p
from nalp.core.dictionary import Dictionary

# Creating a character Dictionary from file
d = Dictionary(from_file='data/text/chapter1_harry.txt', type='char')

# Creating a word Dictionary from file
# d = Dictionary(from_file='data/text/chapter1_harry.txt', type='word')

# Accessing Dictionary properties
print(d.tokens)
print(d.vocab, d.vocab_size)
print(d.vocab_index, d.index_vocab)