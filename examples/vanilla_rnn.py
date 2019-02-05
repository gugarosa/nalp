from nalp.neural.rnn import RNN
from nalp.data.one_hot import OneHotDataset
import nalp.stream.preprocess as p
import tensorflow as tf
import numpy as np

sentences = "i like dog and a cat"

# Creates a pre-processing pipeline
pipe = p.pipeline(
    p.lower_case,
    p.valid_char,
    p.tokenize_to_char
)

# Applying pre-processing pipeline to X
sentences = pipe(sentences)

d = OneHotDataset(sentences, 3)

print(d.inputs)
print(d.targets)


# tf.reset_default_graph()

# sentences = [ "i like dog", "i love coffee", "i hate milk"]

# word_list = " ".join(sentences).split()
# word_list = list(set(word_list))
# word_dict = {w: i for i, w in enumerate(word_list)}
# number_dict = {i: w for i, w in enumerate(word_list)}
# n_class = len(word_dict)

# # TextRNN Parameter
# n_step = 2 # number of cells(= number of Step)
# n_hidden = 5 # number of hidden units in one cell

# def make_batch(sentences):
#     input_batch = []
#     target_batch = []
    
#     for sen in sentences:
#         word = sen.split()
#         input = [word_dict[n] for n in word[:-1]]
#         target = word_dict[word[-1]]

#         input_batch.append(np.eye(n_class)[input])
#         target_batch.append(np.eye(n_class)[target])

#     return input_batch, target_batch

# input_batch, target_batch = make_batch(sentences)

# print(n_class)
rnn = RNN()
rnn.train(d.inputs, d.targets)

print(d.inputs[0:1])
print(d.targets[0:1])

rnn.predict(d.inputs[0:1])