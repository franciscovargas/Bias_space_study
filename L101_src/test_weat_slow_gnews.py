import numpy as np
from WEAT.weat_slow import WEATTest
from WEAT.weat_list import WEATLists
from L101_utils.mock_model import MockModel
from L101_utils.data_paths import googlew2v
from os import path

def main():

    nlp = MockModel.from_file(googlew2v)
    # print("test: {0}".format(nlp("test")))

    test = WEATTest(nlp)
    print(
        test.run_test(
            WEATTest.instruments,
            WEATTest.weapons,
            WEATTest.pleasant,
            WEATTest.unpleasant
        )
    )

if __name__ == '__main__':
    main()
