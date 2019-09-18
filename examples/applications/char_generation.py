import tensorflow as tf

import nalp.utils.preprocess as p
from nalp.corpus.text import TextCorpus
from nalp.datasets.next import NextDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.rnn import RNN

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating next target Dataset
dataset = NextDataset(encoded_tokens, max_length=10, batch_size=64)

# Creating the RNN
rnn = RNN(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# Compiling the RNN
rnn.compile(optimize=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the RNN
rnn.fit(dataset.batches, epochs=250)

# Defining an start string to generate the text
start_string = 'Mr.'

# Generating artificial text
text = rnn.generate_text(encoder, start=start_string, length=1000, temperature=0.5)

# Outputting the text
print(start_string + ''.join(text))