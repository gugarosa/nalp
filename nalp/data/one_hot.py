import numpy as np

class OneHotDataset:
    def __init__(self, tokens, length):
        vocab = list(set(tokens))
        print(vocab)

        self._vocab_size = len(vocab)
        print(self._vocab_size)

        self._char2index = {c: i for i, c in enumerate(vocab)}
        print(self._char2index)

        self._index2char = {i: c for i, c in enumerate(vocab)}
        print(self._index2char)

        self._tokens = tokens
        print(self._tokens)

        self._data = np.array([self._char2index[c] for c in tokens])
        print(self._data)

        self.inputs, self.targets = self.create_examples(self._data, length, self._index2char, self._vocab_size)

    def create_examples(self, data, length, char_index, vocab_size):
        examples = []
        targets = []
        for i in range(0, len(data)-length):
            examples.append(data[i:i+length])
            targets.append(data[i+length])
        
        print(examples)
        print(targets)


        x = np.zeros((len(examples), length, vocab_size), dtype=np.bool)
        y = np.zeros((len(examples), vocab_size), dtype=np.bool)
        for i, sentence in enumerate(examples):
            for t, char in enumerate(sentence):
                x[i, t, char] = 1
            y[i, targets[i]] = 1

        return x, y

