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
    X = ["bagpipe", "cello", "guitar", "lute", "trombone", "banjo", "clarinet", "harmonica", "mandolin", "trumpet", "bassoon", "drum", "harp", "oboe", "tuba", "bell", "fiddle", "harpsichord", "piano", "viola", "bongo",
    "flute", "horn", "saxophone", "violin"]
    # Weapons
    Y = ["arrow", "club", "gun", "missile", "spear", "axe", "dagger", "harpoon", "pistol", "sword", "blade", "dynamite", "hatchet", "rifle", "tank", "bomb", "firearm", "knife", "shotgun", "teargas", "cannon", "grenade",
        "mace", "slingshot", "whip"]
    # Pleasant
    A = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family",
        "happy", "laughter", "paradise", "vacation"]
    # Unpleasant
    B = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", "rotten",
        "vomit", "agony", "prison"]

    print('WEAT d = ', weat.weat_effect_size(X, Y, A, B, emb))
    print('WEAT p = ', weat.weat_p_value(X, Y, A, B, emb, 1000))


if __name__ == '__main__':
    main()
