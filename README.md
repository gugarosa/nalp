# NALP: Natural Adversarial Language Processing

[![Latest release](https://img.shields.io/github/release/gugarosa/nalp.svg)](https://github.com/gugarosa/nalp/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/nalp.svg)](https://github.com/gugarosa/nalp/issues)
[![License](https://img.shields.io/github/license/gugarosa/nalp.svg)](https://github.com/gugarosa/nalp/blob/main/LICENSE)

## Welcome to NALP.

Have you ever wanted to create natural text from raw sources? If yes, NALP is for you! This package is an innovative way of dealing with natural language processing and adversarial learning. From bottom to top, from embeddings to neural networks, we will foster all research related to this new trend.

Use NALP if you need a library or wish to:

* Create your embeddings;
* Design or use pre-loaded state-of-art neural networks;
* Mix-and-match different strategies to solve your problem;
* Because it is cool to play with text.

Read the docs at [nalp.readthedocs.io](https://nalp.readthedocs.io).

NALP 3.0.1 requires **Python 3.11+** and is tested on Python 3.11 through 3.13.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Getting started: 60 seconds with NALP

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think of.

Alternatively, if you wish to learn even more, please take a minute:

NALP is based on the following structure, and you should pay attention to its tree:

```yaml
- nalp
    - core
        - corpus
        - dataset
        - encoder
        - model
    - corpus
        - audio
        - sentence
        - text
    - datasets
        - image
        - language_modeling
    - encoders
        - integer
        - word2vec
    - models
        - discriminators
            - conv
            - embedded_text
            - linear
            - lstm
            - text
        - generators
            - bi_lstm
            - conv
            - gru
            - gumbel_lstm
            - gumbel_rmc
            - linear
            - lstm
            - rmc
            - rnn
            - stacked_rnn
        - layers
            - gumbel_softmax
            - multi_head_attention
            - relational_memory_cell
        - dcgan
        - gan
        - gsgan
        - maligan
        - relgan
        - seqgan
        - wgan
    - utils
        - constants
        - loader
        - logging
        - preprocess
```

### Core

The core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Corpus

Every pipeline has its first step, right? The corpus package serves as a basic class to load raw text, audio and sentences.

### Datasets

Because we need data, right? Datasets are composed of classes and methods that allow preparing data for further neural networks.

### Encoders

Text or Numbers? Encodings are used to make embeddings. Embeddings are used to feed into neural networks. Remember that networks cannot read raw text; therefore, you might want to pre-encode your data using well-known encoders.

### Models

Each neural network architecture is defined in this package. From naïve RNNs to BiLSTMs, from GANs to TextGANs, you can use whatever suits your needs.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use it as you wish than re-implementing the same thing over and over again.

---

## Installation

NALP is published on PyPI. Add it to a project managed by uv with:

```bash
uv add nalp
```

For a consumer installation in an existing Python environment, pip is also supported:

```bash
pip install nalp
```

---

## Development

Clone the repository, then use the locked uv environment for all project tasks:

```bash
git clone https://github.com/gugarosa/nalp.git
cd nalp
uv sync --locked
uv run pytest
uv run pre-commit run --all-files
uv build
```

After a version bump is merged into `main` and CI succeeds, the release workflow
builds and checks the distributions, publishes them to PyPI, and creates the
matching GitHub release. Commits whose version tag already exists do not publish
another release. Publishing a GitHub release manually remains supported.

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
