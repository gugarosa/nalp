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

        self.create_examples(self._data, length)

    def create_examples(self, data, length):
        n_examples = len(data) // length
        print(n_examples)

        examples = []
        targets = []
        for i in range(0, n_examples - length):
            examples.append(data[i:i+length])
            targets.append(data[i+length])
        
        print(examples)
        print(targets)

