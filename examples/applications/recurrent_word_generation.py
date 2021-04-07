from nalp.corpus import TextCorpus
from nalp.encoders import IntegerEncoder
from nalp.models.generators import RNNGenerator

# When generating artificial text, make sure
# to use the same data, classes and parameters
# as the pre-trained network

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='word')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Creating the RNN
rnn = RNNGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# Loading pre-trained RNN weights
rnn.load_weights('trained/rnn').expect_partial()

# Now, for the inference step, we build with a batch size equals to 1
rnn.build((1, None))

# Defining an start string to generate the text
start_string = 'Mr. and Mrs. Dursley'

# Generating artificial text
text = rnn.generate_temperature_sampling(start=start_string.split(' '), max_length=1000, temperature=0.5)

# Outputting the text
print(start_string + ' ' + ' '.join(text))
