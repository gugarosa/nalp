Public API contracts
====================

Constructor arguments are documented on ``__init__``. Keras-dispatched hooks use
the framework's interfaces rather than repeating generic method docstrings.
The NALP-specific data, state, and ownership contracts are collected here.

Corpora and encoders
--------------------

Concrete corpora populate the following public state inherited from
:class:`nalp.core.corpus.Corpus`:

``tokens``
    A flat token list for text and audio, or a list of token lists for sentences.
    Nonempty lists supplied by the caller are reused and may be modified during
    frequency filtering, padding, and truncation.

``vocab``
    The sorted unique vocabulary, including the unknown-token marker.

``vocab_size``
    The number of entries in ``vocab``.

``vocab_index``
    The token-to-integer mapping produced by enumerating ``vocab``.

``index_vocab``
    The inverse integer-to-token mapping.

``min_frequency``
    The corpus-wide count threshold used when replacing infrequent tokens.

:class:`nalp.encoders.integer.IntegerEncoder` retains its ``encoder`` and
``decoder`` dictionaries by reference. ``learn()`` binds those mappings rather
than copying them. Caller mutations therefore affect subsequent operations.
Unknown tokens use the ``<UNK>`` entry when it exists, while decoding an absent
ID raises ``KeyError``.

Integer encoding produces ``int32`` arrays and preserves a flat or rectangular
nested structure. A character prompt can be a string. A word prompt must supply
the already-tokenized words rather than relying on character-by-character string
iteration.

:class:`nalp.encoders.word2vec.Word2vecEncoder` stores a learned Gensim Word2Vec
model in ``encoder``. Its vector lookup returns a ``float64`` NumPy array, and
decoding selects the most similar learned token for each vector.

Datasets
--------

:class:`nalp.core.dataset.Dataset` is a NALP wrapper, not a TensorFlow dataset.
Pass its ``batches`` attribute to training methods. Before a concrete pipeline
has been built, ``batches`` is ``None``.

``shuffle`` controls sample shuffling during pipeline construction. Changing
that attribute afterward does not rebuild an existing pipeline. The current
batching policy drops incomplete final batches.

:class:`nalp.datasets.image.ImageDataset` copies images to ``float32`` and can
reshape them before batching. Normalization assumes pixel values in the range
0 through 255 and maps them to the range -1 through 1.

:class:`nalp.datasets.language_modeling.LanguageModelingDataset` emits
``(inputs, targets)`` pairs. A one-dimensional token stream is split into
nonoverlapping windows of input length plus one. A rectangular sentence array
is already grouped into sequences. Each sequence becomes ``sequence[:-1]`` and
``sequence[1:]`` without changing token-ID dtype.

Model inputs and outputs
------------------------

Let ``B`` denote batch size, ``T`` timesteps, ``V`` vocabulary size, and ``F`` a
feature count. Model input and output shapes depend on the component family:

Ordinary recurrent generators
    Integer token IDs shaped ``(B, T)`` produce floating logits shaped
    ``(B, T, V)``. These are scores before softmax, not probabilities.

Gumbel recurrent generators
    Integer token IDs shaped ``(B, T)`` produce a tuple of logits, relaxed
    probabilities, and sampled token IDs. The first two tensors have shape
    ``(B, T, V)``, and the integer IDs have shape ``(B, T)``.

Linear image components
    Generators consume noise with final dimension ``noise_dim`` and preserve
    its leading dimensions while producing the configured feature count.
    Discriminators preserve leading dimensions and produce one final logit.

Convolutional image components
    Images use channels-last layout. Adversarial sampling supplies noise shaped
    ``(B, 1, 1, noise_dim)``. Discriminator outputs may retain spatial dimensions,
    so callers must not assume every discriminator returns shape ``(B, 1)``.

Embedded text discriminator
    Token IDs shaped ``(B, T)`` produce logits shaped ``(B, 1, 2)`` when the
    configured sequence length matches the input.

Text discriminator
    Vocabulary vectors shaped ``(B, T, V)`` produce a feature tensor shaped
    ``(B, 1, sum(n_filters))`` for the configured sequence length.

LSTM discriminator
    Vocabulary vectors shaped ``(B, T, V)`` produce logits shaped ``(B, T, 1)``.

The default floating computation policy is ``float32``. This convention
migration does not add alternative dtype-policy support or change model math.

State and training ownership
----------------------------

Recurrent generators retain state between calls. Their batch size is fixed
when recurrent state is built. Create a new model instance and load compatible
weights when moving from a training batch size to an inference batch size.
Call ``reset_state()`` or ``reset_states()`` before an independent sequence.
Relational-memory generators restore the cell's identity-based initial memory.

All token-generation methods select from raw logits, including Gumbel
generators. They reset state before sampling, exclude the prompt from the
returned tokens, and include ``<EOS>`` if it terminates generation.
Gumbel temperature sampling also updates the public ``tau`` setting.
Temperature remains runtime configuration rather than an additional persisted
Keras weight.

:class:`nalp.core.model.Adversarial` retains components as ``D`` and ``G``.
``compile()`` configures ``D_optimizer`` and ``G_optimizer``, creates the
``D_loss`` and ``G_loss`` metrics, and initializes their series in ``history``.
Text adversarial trainers also own ``P_optimizer`` and corresponding
pretraining history series.

NALP's custom ``fit()`` and ``pre_fit()`` methods return ``None`` and update
``history``. They are not replacements for the full Keras ``fit()`` interface.
Ordinary recurrent generators still use native Keras training.

Materialization and persistence
-------------------------------

A representative forward call materializes lazy component weights. A
``built`` flag alone does not establish that a custom model's children have
created all their weights. Warm-up calls can change recurrent state, so reset
that state before independent inference or output comparisons.

Keep component weights, architecture configuration, encoder mappings, and
training-resume state conceptually separate. Retaining the exact token-ID
mapping is necessary for meaningful generation after loading weights.
Current Keras weights-only files use ``.weights.h5``.

Historical examples using ``save_format="tf"`` or ``expect_partial()`` require
a separate checkpoint migration. This convention change does not reinterpret
existing checkpoints, promise whole-GAN serialization, or guarantee complete
training resumption.

Logging
-------

Use :func:`nalp.utils.logging.get_logger` for explicit NALP logging.
:meth:`nalp.utils.logging.Logger.to_file` emits only through configured file
handlers, honors logger and handler filters, and leaves console-handler levels
unchanged. Handler owners are responsible for closing their handlers.
