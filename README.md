# NALP: Natural Adversarial Language Processing

## Welcome to NALP.
Have you ever wanted to created natural text from raw sources? If yes, NALP is for you! This package is an innovative way of dealing with natural language processing and adversarial learning. From bottom to top, from embeddings to neural networks, we will foster all research related to this newly trend.

Use NALP if you need a library or wish to:
* Create your own embeddings.
* Design or use pre-loaded state-of-art neural networks.
* Mix-and-match different strategies to solve your problem.
* Because it is cool to play with text.

Read the docs at [nalp.recogna.tech](http://nalp.recogna.tech).

NALP is compatible with: **Python 2.7-3.6**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy, if you wish to read the code and bump yourself into, just follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.



---

## Getting started: 60 seconds with NALP

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage and follow the example. We have high-level examples for most tasks we could think of.

Or if you wish to learn even more, please take a minute:

NALP is based on the following structure, and you should pay attention to its tree:

```
- nalp
    - core
        - dataset
        - encoder
        - neural
    - datasets
    - encoders
    - neurals
    - stream
    - utils
    - visualization
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basic of our structure. They should provide variables and methods that will help to construct other modules. It is composed by the following classes:

1. Dataset (used to handle receiving (can be raw of pre-processed) data and preparing it for further neural package methods)

2. Encoder (You can use different pre-stablished encoders as well. They should provide that matrix you were wanting all the time. For example: CountVectorizer, TF-IDF and Word2Vec.)

3. Neural (This is the brain of the system. It will hold all the high-level methods in order to interact directly from tensorflow. That is it. No Keras. We use tensorflow. RAW. We like to learn and believe that machine learning is mathematics. TF proved to be great for us.)

### Datasets

### Encoders

### Neurals

### Stream

### Utils

### Visualization

---

## Installation

We belive that everything have to be easy. Not diffucult or daunting, NALP will be the one-to-go package that you will need, from the very first instalattion to the daily-tasks implementing needs. If you may, just run the following under your most preferende Python environment (raw, conda, virtualenv, whatever)!:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for an additional implementation. If needed, from here you will be the one to know all of its details.

### Ubuntu

No specific additional commandad needed.

### Windows

No specific additional commandad needed.

### MacOS

No specific additional commandad needed.

---

## Support

We know that we do our best, but it's inevitable to acknowlodge that we make mistakes. If you every need to report a bug, report a problem, talk to us, please do so! We will be avaliable at our bests at this repository or recogna@fc.unesp.br.

---
