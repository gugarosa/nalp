# NALP: Natural Adversarial Language Processing

[![Latest release](https://img.shields.io/github/release/gugarosa/nalp.svg)](https://github.com/gugarosa/nalp/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/nalp.svg)](https://github.com/gugarosa/nalp/issues)
[![License](https://img.shields.io/github/license/gugarosa/nalp.svg)](https://github.com/gugarosa/nalp/blob/master/LICENSE)

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
        - one_hot
        - vanilla
    - encoders
        - count
        - tfidf
        - word2vec
    - neurals
        - rnn
    - stream
        - loader
        - preprocess
    - utils
        - decorators
        - logging
        - splitters
    - visualization
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basic of our structure. They should provide variables and methods that will help to construct other modules. It is composed by the following classes:

```dataset```: Used to handle receiving (can be raw of pre-processed) data and preparing it for further neural package methods.

```encoder```: You can use different pre-stablished encoders as well. They should provide that matrix you were wanting all the time. For example: CountVectorizer, TF-IDF and Word2Vec.

```neural```: This is the brain of the system. It will hold all the high-level methods in order to interact directly from tensorflow. That is it. No Keras. We use tensorflow. RAW. We like to learn and believe that machine learning is mathematics. TF proved to be great for us.

### Datasets

Because we need data, right? Datasets are composed by classes and methods that allow to prepare data for further neural networks.

```one_hot```: An one hot encoding for dataset. This serves as a basis for predicting next characters or words.

```vanilla```: A vanilla dataset, used to load (X, Y) sample pairs. One can see as a basic structure for common datasets, where each sample is composed by features and labels.

### Encoders

Text or Numbers? Encodings are used to make embeddings. Embeddings are used to feed into neural networks. Remember that networks cannot read raw data, therefore you might want to pre-encode your data using well-known encoders.

```count```: CountVectorizer encoding.

```tfidf```: TF-IDF encoding.

```word2vec```: Word2Vec encoding.

### Neurals

A neural networks package. In this package you can find all neural-related implementations. From naïve RNNs to BiLSTMs, you can use whatever suits your needs. All implementations were done using raw Tensorflow, mainly to better understand and control the whole training and inference process.

```rnn```: A naïve Recurrent Neural Network implementation.

### Stream

A stream package is used to manipulate data. From loading to processing, here you can find all classes and methods defined in order to help you achieve these tasks.

```loader```: Loading module, used to load external text data.

```preprocess```: Pre-processing module, used to pre-process, tokenize and many other methods.

### Utils

This is an utilities package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing over and over again.

```decorators```: Decorators used to facilitate repetitive declarations.

```logging```: Logging tools to track the progress of a NALP task.

```splitters```: A data splitter tool. It might be removed in the future.

### Visualization

A visualization package in order to better illustrate what is happening with your data. Use classes and methods to help you decide if your data is well enough to fulfill your desires.

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

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it's inevitable to acknowlodge that we make mistakes. If you every need to report a bug, report a problem, talk to us, please do so! We will be avaliable at our bests at this repository or recogna@unesp.br.

---
