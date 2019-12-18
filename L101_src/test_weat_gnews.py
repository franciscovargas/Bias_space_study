import numpy as np
import gensim
from WEAT.weat_slow import WEATTest
from WEAT.weat_list import WEATLists
import WEAT.weat as weat
from L101_utils.mock_model import MockModel
from L101_utils.data_paths import googlew2v

def main():
    emb = MockModel.from_file(googlew2v, mock=False)

    # Instruments
    X = WEATLists.W_2_Instruments
    # Weapons
    Y = WEATLists.W_2_Weapons
    # Pleasant
    A = WEATLists.W_2_Pleasant
    # Unpleasant
    B = WEATLists.W_2_Unpleasant

    print('WEAT d = ', weat.weat_effect_size(X, Y, A, B, emb))
    print('WEAT p = ', weat.weat_p_value(X, Y, A, B, emb, 10000))


if __name__ == '__main__':
    main()
