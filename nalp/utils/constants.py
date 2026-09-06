# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Define shared token markers and numerical controls.

Attributes:
    SOS: Start-of-sentence marker.
    EOS: End-of-sentence marker.
    PAD: Sentence-padding marker.
    UNK: Unknown-token marker.
    BUFFER_SIZE: Maximum number of dataset elements buffered during shuffling.
    D_STEPS: Number of discriminator minibatch updates per selected batch.
    EPSILON: Small positive offset for numerically sensitive expressions.

"""

SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"

BUFFER_SIZE = 100000

D_STEPS = 3

EPSILON = 1e-20
