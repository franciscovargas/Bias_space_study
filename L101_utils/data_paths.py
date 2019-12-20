from os import path


root = path.dirname(path.abspath(path.dirname(__file__)))

data = path.join(root, "data")
data_bolu = path.join(data, "bolukbasi_data")

googlew2v = path.join(data, "GoogleNews-vectors-negative300.bin")
googlew2vtxt = path.join(data, "GoogleNews-vectors-negative300.txt")
wikift = path.join(data, "wiki-news-300d-1M.vec")

bolu_googlew2v = path.join(data_bolu, "GoogleNews-vectors-negative300-hard-debiased.bin")

bolu_gender_seed = path.join(data_bolu, "gender_specific_seed.json")
bolu_gender_specific = path.join(data_bolu, "gender_specific_full.json")
bolu_equalize_pairs = path.join(data_bolu, "equalize_pairs.json")
bolu_definitional_pairs = path.join(data_bolu, "definitional_pairs.json")
