import OpenHowNet
from textattack.transformations import WordSwap


class ChineseWordSwapHowNet(WordSwap):
    def __init__(self, topk=5):
        self.hownet_dict = OpenHowNet.HowNetDict(init_sim=True)
        self.topk = topk

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with N characters
        replaced by a homoglyph."""
        results = self.hownet_dict.get_nearest_words(word, language="zh", K=self.topk)
        synonyms = set()
        if results:
            for key, value in results.items():
                for w in value:
                    synonyms.add(w)
            return list(synonyms)
        else:
            return []