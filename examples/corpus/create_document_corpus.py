import nalp.utils.preprocess as p
from nalp.corpus.document import DocumentCorpus

# Creating a character DocumentCorpus from file
corpus = DocumentCorpus(from_file='data/document/chapter1_harry.txt')

# Accessing DocumentCorpus properties
print(corpus.tokens)
