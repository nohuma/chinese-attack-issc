import nltk
import textattack
from nltk.corpus import wordnet as wn

from textattack.transformations import WordSwap
# nltk.download('wordnet')
# nltk.download('omw')


class ChineseWordSwapWordNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    WordNet.

    >>> from textattack.transformations import WordSwapWordNet
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapWordNet()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, language="eng"):
        wn._load_lang_data(language)
        if language not in wn.langs():
            raise ValueError(f"Language {language} not one of {wn.langs()}")
        self.language = language

    def _get_replacement_words(self, word, random=False):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        synonyms = set()
        for syn in wn.synsets(word, lang=self.language):
            if syn is None:
                continue
            for syn_word in syn.lemma_names(lang=self.language):
                if (
                    (syn_word != word)
                    and ("_" not in syn_word)
                    and (textattack.shared.utils.is_one_word(syn_word))
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.add(syn_word)
        return list(synonyms)

def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True

    return False
