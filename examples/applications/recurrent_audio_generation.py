import numpy as np
import tensorflow as tf
from mido import Message, MidiFile, MidiTrack

from nalp.corpus import AudioCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models.generators import RNNGenerator

# Creating an AudioCorpus from file
corpus = AudioCorpus(from_file="data/audio/sample.mid")

# Creating an IntegerEncoder, learning encoding and encoding tokens
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(
    encoded_tokens, max_contiguous_pad_length=100, batch_size=64
)

# Creating the RNN
rnn = RNNGenerator(
    encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512
)

# As NALP's RNNs are stateful, we need to build it with a fixed batch size
rnn.build((64, None))

# Compiling the RNN
rnn.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# Fitting the RNN
rnn.fit(dataset.batches, epochs=25)

# Saving RNN weights
rnn.save_weights("trained/audio_rnn", save_format="tf")

# Re-creating the RNN
rnn = RNNGenerator(
    encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512
)

# Loading pre-trained RNN weights
rnn.load_weights("trained/audio_rnn").expect_partial()

# Now, for the inference step, we build with a batch size equals to 1
rnn.build((1, None))

# Generating artificial notes
notes = rnn.generate_temperature_sampling(start=[55], max_length=1000, temperature=0.5)

# Creating midi classes to hold generated audio and further music track
audio = MidiFile()
track = MidiTrack()

# Creating a time counter
t = 0

# Iterating through generated notes
for note in notes:
    # Creating a note array
    note = np.asarray([147, note, 67])

    # Converting to bytes
    bytes = note.astype(int)

    # Gathering a step
    step = Message.from_bytes(bytes[0:3])

    # Increasing track counter
    t += 1

    # Applying current time as current step
    step.time = t

    # Appending to track
    track.append(step)

# Appending track to file
audio.tracks.append(track)

# Outputting generated .midi file
audio.save("generated_sample.mid")
