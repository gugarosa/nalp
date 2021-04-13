from nalp.corpus import TextCorpus
from nalp.encoders import IntegerEncoder
from nalp.models import SeqGAN

# When generating artificial text, make sure
# to use the same data, classes and parameters
# as the pre-trained network

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='char')

# Creating an IntegerEncoder and learning encoding
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Creating the SeqGAN
seqgan = SeqGAN(encoder=encoder, vocab_size=corpus.vocab_size, max_length=10, embedding_size=256,
                hidden_size=512, n_filters=(64, 128, 256), filters_size=(3, 5, 5), dropout_rate=0.25, temperature=1)

# Loading pre-trained SeqGAN weights
seqgan.load_weights('trained/seqgan').expect_partial()

# Now, for the inference step, we build with a batch size equals to 1
seqgan.G.build((1, None))

# Defining an start string to generate the text
start_string = 'Mr.'

# Generating artificial text
text = seqgan.G.generate_temperature_sampling(start=start_string, max_length=1000, temperature=1)

# Outputting the text
print(start_string + ''.join(text))
