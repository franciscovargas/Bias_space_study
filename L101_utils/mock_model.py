import numpy as np
from collections import OrderedDict
import gensim
import warnings


class MockModel(object):

    @classmethod
    def from_file(cls, file_name, mock=True):
        *name, extension = file_name.split(".")
        print(name, extension)
        bin = True if "bin" in extension else False

        mod = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=bin)

        if mock:
            return cls(mod.wv.index2word, mod.wv.vectors)
        else:
            return mod

    def __init__(self, index2word, X):
        self.key  = OrderedDict(zip(index2word,
                                    list(range(len(index2word)))))
        self.X = X
        self.n, self.dim = X.shape
        self.index2word = index2word

    def __call__(self, w):
        return self.get_word_vector(w)

    def get_word_vector(self, w):
        if w in self.key:
            return self.X[self.key[w], :]
        elif w.lower() in self.key:
            return self.X[self.key[w.lower()], :]
        elif w.lower().strip() in self.key:
            return self.X[self.key[w.lower().strip()], :]
        else:
            print("cake: ", w)
            warnings.warn(f"Word {w} not in model", RuntimeWarning)
            return np.zeros(self.dim) + 1e-10
