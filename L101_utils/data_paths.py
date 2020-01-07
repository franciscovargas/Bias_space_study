from os import path


root = path.dirname(path.abspath(path.dirname(__file__)))

model = path.join(root, "models")
data = path.join(root, "data")
data_bolu = path.join(data, "bolukbasi_data")
data_gonen = path.join(data, "gonen_data")


male_words = path.join(data_gonen, "male_words_file.txt")
female_words = path.join(data_gonen, "female_words_file.txt")


googlew2v = path.join(data, "GoogleNews-vectors-negative300.bin")
googlew2vtxt = path.join(data, "GoogleNews-vectors-negative300.txt")
glove = path.join(data, "glove.txt")
smallc_glove = path.join(data, "glove_small_complete.bin")
small_googlew2v = path.join(data, "google_small.bin")
wikift = path.join(data, "wiki-news-300d-1M.vec")

bolu_googlew2v = path.join(data_bolu, "GoogleNews-vectors-negative300-hard-debiased.bin")

bolu_gender_seed = path.join(data_bolu, "gender_specific_seed.json")
bolu_gender_specific = path.join(data_bolu, "gender_specific_full.json")
bolu_equalize_pairs = path.join(data_bolu, "equalize_pairs.json")
bolu_definitional_pairs = path.join(data_bolu, "definitional_pairs.json")
bolu_professions = path.join(data_bolu, "professions.json")

my_linear_debias = path.join(data, "my_weat_linear_debias_vectors.bin")
my_linear_debias_k_2 = path.join(data, "my_weat_linear_debias_vectors_k_2.bin")
my_kpca_debias_k_1 = path.join(data, "my_weat_kpca_debias_vectors_k_1.bin")
