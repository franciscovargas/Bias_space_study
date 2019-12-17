from os import path


root = path.dirname(path.abspath(path.dirname(__file__)))

data = path.join(root, "data")

googlew2v = path.join(data, "GoogleNews-vectors-negative300.bin")
wikift = path.join(data, "wiki-news-300d-1M.vec")
