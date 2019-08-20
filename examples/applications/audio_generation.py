import mido
import numpy as np
from mido import Message, MidiFile, MidiTrack

import nalp.stream.loader as l
import nalp.stream.preprocess as p
from nalp.datasets.one_hot import OneHot
from nalp.neurals.rnn import RNN

# Loading .midi file
audio = MidiFile('data/audio/sample.mid')

# Declaring an empty list to hold audio notes
notes = []

# Gathering notes
for step in audio:
    # Checking for real note
    if not step.is_meta and step.channel == 0 and step.type == 'note_on':
        # Gathering note
        note = step.bytes()

        # Saving to string
        notes.append(note[1])

# Creating a OneHot dataset
d = OneHot(notes, max_length=250)

# Defining a neural network based on vanilla RNN
rnn = RNN(vocab_size=d.vocab_size, hidden_size=128, learning_rate=0.001)

# Training the network
rnn.train(train=d, batch_size=128, epochs=5)

# Generating new notes
gen_notes = rnn.generate_text(
    dataset=d, start_text=[55], length=1000, temperature=0.2)

# Creating midi classes to hold generated audio and further music track
new_audio = MidiFile()
track = MidiTrack()

# Creating a time counter
t = 0

# Iterating through generated notes
for note in gen_notes:
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
new_audio.tracks.append(track)

# Outputting generated .midi file
new_audio.save('generated_sample.mid')
