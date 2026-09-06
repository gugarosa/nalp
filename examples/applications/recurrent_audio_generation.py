# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import tensorflow as tf
from mido import Message, MidiFile, MidiTrack

from nalp.corpus import AudioCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models.generators import RNNGenerator

corpus = AudioCorpus(from_file="data/audio/sample.mid")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=100, batch_size=64)

rnn = RNNGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# Stateful recurrent models require a fixed training batch size
rnn.build((64, None))
rnn.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

rnn.fit(dataset.batches, epochs=25)

rnn.save_weights("trained/audio_rnn", save_format="tf")

# Recreate the stateful model for single-sequence inference
rnn = RNNGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

rnn.load_weights("trained/audio_rnn").expect_partial()

# Inference uses a single sequence per batch
rnn.build((1, None))
notes = rnn.generate_temperature_sampling(start=[55], max_length=1000, temperature=0.5)

audio = MidiFile()
track = MidiTrack()
t = 0
for note in notes:
    note = np.asarray([147, note, 67])
    bytes = note.astype(int)
    step = Message.from_bytes(bytes[0:3])
    t += 1
    step.time = t
    track.append(step)

audio.tracks.append(track)
audio.save("generated_sample.mid")
