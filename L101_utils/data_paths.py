from os import path


root = path.dirname(path.abspath(path.dirname(__file__)))

data = path.join(root, "data")
data_bolb = path.join(data, "bolukbasi_data")

googlew2v = path.join(data, "GoogleNews-vectors-negative300.bin")
wikift = path.join(data, "wiki-news-300d-1M.vec")

bolu_googlew2v = path.join(data_bolb, "GoogleNews-vectors-negative300-hard-debiased.bin")
