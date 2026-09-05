Welcome to NALP's documentation!
=================================

Have you ever wanted to create natural text from raw sources? If yes, NALP is for you! This package is an innovative way of dealing with natural language processing and adversarial learning. From bottom to top, from embeddings to neural networks, we will foster all research related to this new trend.

Use NALP if you need a library or wish to:

* Create your embeddings;
* Design or use pre-loaded state-of-art neural networks;
* Mix-and-match different strategies to solve your problem;
* Because it is cool to play with text.

NALP requires **Python 3.11+** and is tested on Python 3.11 through 3.13.

Generation and training contracts
=================================

Recurrent generators keep their state between calls. Call ``reset_state()``
(or ``reset_states()``) before starting an independent sequence. Relational
memory generators restore the cell's identity-based initial memory on reset.
The batch size is fixed when stateful recurrent layers are built; use a new
instance and load its weights when switching to a different batch size.

All recurrent generators, including Gumbel variants, use raw logits for greedy,
temperature, and top-k/top-p generation. Greedy search takes the largest logit.
Temperature must be finite and positive. Top-p retains the smallest sorted
prefix whose cumulative probability reaches the requested threshold; ``p=0``
disables this filter. Gumbel generators still return
``(logits, relaxed_probabilities, sampled_tokens)`` from a normal forward call.
Assigning ``generator.tau`` updates already-traced training steps; temperature
remains runtime configuration rather than part of a Keras weights file.

``LanguageModelingDataset`` provides input context and next-token targets.
MaliGAN and SeqGAN train their discriminators and calculate rewards on the
targets, not the initial context token. Their discriminator labels are
``0`` for real sequences and ``1`` for generated sequences. MaliGAN normalizes
real-class odds across the batch; SeqGAN averages independent continuations
of each generated prefix without mixing samples or timesteps.

.. toctree::
    :maxdepth: 2
    :caption: Package Reference

    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
