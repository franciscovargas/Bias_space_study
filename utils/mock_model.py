import numpy as np
from collections import OrderedDict
import gensim
import warnings


class MockModel(object):

    @classmethod
    def from_file(cls, file_name):
        name, extension = file_name.split(".")
        bin = True if "bin" in extension else False

        mod = gensim.models.Word2Vec.load_word2vec_format(file_name, binary=bin)

        return cls(mod.wv.index2word, mod.wv.vectors)

    def __init__(self, word_list, X):
        self.key  = OrderedDict(index2word, zip(list(range(len(word_list)))))
        self.X = X
        self.n, self.dim = X.shape
        self.index2word = index2word

    def __call__(self, w):
        return self.get_word_vector(w)

    def get_word_vector(self, w):
        if w in self.key:
            return self.X[self.key[w], :]
        elif w.lower() in in self.key:
            return self.X[self.key[w.lower()], :]
        elif w.lower().strip() in self.key:
            return self.X[self.key[w.lower().strip()], :]
        else:
            warnings.warn(f"Word {w} not in model", RuntimeWarning)
            return np.zeros(dim)
