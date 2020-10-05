# Bias_space_study


Code for EMNLP 2020 submission: Exploring the Linear Subspace Hypothesis in Gender Bias Mitigation (https://arxiv.org/pdf/2009.09435.pdf). 

## Instalaltion

In the root directory of the repo run:
```
pip install -e .
```

## Datasets

Standard glove and google news word2vec embeddings must be downloaded (with dimensions as per the paper) and placed in the data forlder.

Gonen and Bolukbasi data provided inside the data folder. 

Hyperpaprameters are set to the default mentione in the paper (default standard calculation provided by sklearn).


## Run

To run a specific model first one must go to the kpca debiase script and set the name of the output model pkl at https://github.com/franciscovargas/Bias_space_study/blob/f0e432ec5115c24d3290e2e4fd52bea61617ed25/L101_src/debiase_kpca.py#L57  then run 

```
python debiase_kpca.py
```

To save the model. To specify the kernel change the flag in this string https://github.com/franciscovargas/Bias_space_study/blob/f0e432ec5115c24d3290e2e4fd52bea61617ed25/L101_src/debiase_kpca.py#L28 . Note gamma is set to the default value by sklearn gamma=None, however the user can experiment with other settings, we found via a small grid search that it barely impacts results.

Once the model has been saved you can edit for the saved models name in a particular task/experiment and obtain results. For example for WEAT : 

* change the pickle model name to your saved model here: https://github.com/franciscovargas/Bias_space_study/blob/f0e432ec5115c24d3290e2e4fd52bea61617ed25/WEAT/weat_test_list.py#L32
* then run:

```
python weat_test_list.py
```

For the other relevant experiments the files are:

* https://github.com/franciscovargas/Bias_space_study/blob/master/L101_src/lipstick_on_a_pig_classification.py
* https://github.com/franciscovargas/Bias_space_study/blob/master/L101_src/lipstick_on_a_pig_professionals.py
* https://github.com/franciscovargas/Bias_space_study/blob/master/L101_src/simlex999_eval.py

Usage is the same as with `weat_test_list.py`  change the name of the file where the model is loaded, then simply run the file. 
