from WEAT.weat_slow import WEATTest
from WEAT.weat_list import WEATLists
from L101_utils.data_paths import bolu_googlew2v
import numpy as np
import WEAT.weat as weat
from L101_utils.mock_model import MockModel
# from L101_utils.data_paths import googlew2v


def w_test(vec_path=None):
    emb = MockModel.from_file(vec_path, mock=False)

    for (X,Y,A,B) in WEATLists.TEST_LIST:
        print('WEAT d = ', weat.weat_effect_size(X, Y, A, B, emb))
        print('WEAT p = ', weat.weat_p_value(X, Y, A, B, emb, 1000))


if __name__ == '__main__':
    w_test(bolu_googlew2v)
