class WEATTest(object):
    """
    Perform WEAT (Word Embedding Association Test) bias tests on a language model.
    Follows from Caliskan et al 2017 (10.1126/science.aal4230).
    
    Taken from: https://gist.github.com/SandyRogers/e5c2e938502a75dcae25216e4fae2da5
    """
    
    instruments = ["bagpipe", "cello", "guitar", "lute", "trombone", "banjo", "clarinet", "harmonica", "mandolin", "trumpet", "bassoon", "drum", "harp", "oboe", "tuba", "bell", "fiddle", "harpsichord", "piano", "viola", "bongo",
"flute", "horn", "saxophone", "violin"]
    weapons = ["arrow", "club", "gun", "missile", "spear", "axe", "dagger", "harpoon", "pistol", "sword", "blade", "dynamite", "hatchet", "rifle", "tank", "bomb", "firearm", "knife", "shotgun", "teargas", "cannon", "grenade",
    "mace", "slingshot", "whip"]
    flowers = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose", "blue-bell", "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet", "carnation", "gladiola", "magnolia", "petunia", "zinnia"]
    insects = ["ant", "caterpillar", "flea", "locust", "spider", "bedbug", "centipede", "fly", "maggot", "tarantula", "bee", "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket", "hornet", "moth", "wasp", "blackfly",
    "dragonfly", "horsefly", "roach", "weevil"]
    pleasant = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family",
    "happy", "laughter", "paradise", "vacation"]
    unpleasant = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", "rotten",
    "vomit", "agony", "prison"]
    math = ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"]
    arts = ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"]
    male = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
    female = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
    science = ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"]
    arts_2 = ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama"]
    male_2 = ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"]
    female_2 = ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "he"]
    mental_disease = ["sad", "hopeless", "gloomy", "tearful", "miserable", "depressed"]
    physical_disease = ["sick", "illness", "influenza", "disease", "virus", "cancer"]
    temporary = ["impermanent", "unstable", "variable", "fleeting", "short-term", "brief", "occasional"]
    permanent = ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"]
    
    def __init__(self, model):
        """Setup a Word Embedding Association Test for a given spaCy language model.
        
        EXAMPLE:
            >>> nlp = spacy.load('en_core_web_md')
            >>> test = WEATTest(nlp)
            >>> test.run_test(WEATTest.instruments, WEATTest.weapon, WEATTest.pleasant, WEATTest.unpleasant)
        """
        self.model = model

    @staticmethod
    def word_association_with_attribute(w, A, B):
        return np.mean([w.similarity(a) for a in A]) - np.mean([w.similarity(b) for b in B])

    @staticmethod
    def differential_assoication(X, Y, A, B):
        return np.sum([word_association_with_attribute(x, A, B) for x in X]) - np.sum([word_association_with_attribute(y, A, B) for y in Y])

    @staticmethod
    def weat_effect_size(X, Y, A, B):
        return (
            np.mean([word_association_with_attribute(x, A, B) for x in X]) -
            np.mean([word_association_with_attribute(y, A, B) for y in Y])
        ) / np.std([word_association_with_attribute(w, A, B) for w in X + Y])

    @staticmethod
    def _random_permutation(iterable, r=None):
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        return tuple(random.sample(pool, r))

    @staticmethod
    def weat_p_value(X, Y, A, B, sample):
        size_of_permutation = min(len(X), len(Y))
        X_Y = X + Y
        observed_test_stats_over_permutations = []

        if not sample:
            permutations = combinations(X_Y, size_of_permutation)
        else:
            permutations = [random_permutation(X_Y, size_of_permutation) for s in range(sample)]

        for Xi in permutations:
            Yi = filterfalse(lambda w: w in Xi, X_Y)
            observed_test_stats_over_permutations.append(differential_assoication(Xi, Yi, A, B))

        unperturbed = differential_assoication(X, Y, A, B)
        is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
        return is_over.sum() / is_over.size

    @staticmethod
    def weat_stats(X, Y, A, B, sample_p=None):
        test_statistic = differential_assoication(X, Y, A, B)
        effect_size = weat_effect_size(X, Y, A, B)
        p = weat_p_value(X, Y, A, B, sample=sample_p)
        return test_statistic, effect_size, p

    def run_test(self, target_1, target_2, attributes_1, attributes_2, sample_p=None):
        """Run the WEAT test for differential association between two 
        sets of target words and two seats of attributes.
        
        EXAMPLE:
            >>> test.run_test(WEATTest.instruments, WEATTest.weapon, WEATTest.pleasant, WEATTest.unpleasant)
            >>> test.run_test(a, b, c, d, sample_p=1000) # use 1000 permutations for p-value calculation
            >>> test.run_test(a, b, c, d, sample_p=None) # use all possible permutations for p-value calculation
            
        RETURNS:
            (d, e, p). A tuple of floats, where d is the WEAT Test statistic, 
            e is the effect size, and p is the one-sided p-value measuring the
            (un)likeliness of the null hypothesis (which is that there is no
            difference in association between the two target word sets and
            the attributes).
            
            If e is large and p small, then differences in the model between 
            the attribute word sets match differences between the targets.
        """
        X = [self.model(w) for w in target_1]
        Y = [self.model(w) for w in target_2]
        A = [self.model(w) for w in attributes_1]
        B = [self.model(w) for w in attributes_2]
        return weat_stats(X, Y, A, B, sample_p)
