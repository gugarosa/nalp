import nalp.utils.preprocess as p
from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_length=10, batch_size=1, shuffle=True)

# Iterating over one batch
for input_batch, target_batch in dataset.batches.take(1):
    # For every input and target inside the batch
    for input, target in zip(input_batch, target_batch):
        # Transforms the tensor to numpy and decodes it
        print(encoder.decode(input.numpy()), encoder.decode(target.numpy()))
