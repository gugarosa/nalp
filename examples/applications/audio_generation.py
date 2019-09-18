import numpy as np
import tensorflow as tf
from mido import Message, MidiFile, MidiTrack

import nalp.utils.preprocess as p
from nalp.corpus.audio import AudioCorpus
from nalp.datasets.next import NextDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.rnn import RNN

# Creating an AudioCorpus from file
corpus = AudioCorpus(from_file='data/audio/sample.mid')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the AudioCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating next target Dataset
dataset = NextDataset(encoded_tokens, max_length=100, batch_size=64)

# Creating the RNN
rnn = RNN(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# Compiling the RNN
rnn.compile(optimize=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the RNN
rnn.fit(dataset.batches, epochs=25)

# Generating artificial notes
notes = rnn.generate_text(encoder, start=[55], length=1000, temperature=0.2)

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
audio.save('generated_sample.mid')
