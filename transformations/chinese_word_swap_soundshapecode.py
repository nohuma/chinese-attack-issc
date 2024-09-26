from textattack.transformations import WordSwap
from soundshapecode import *


class ChineseSSCCharacterSwap(WordSwap):
    def __init__(self, encode_way='ALL', num_words=5) -> None:
        getHanziSSCDict()
        self.encode_way = encode_way
        self.num_words = num_words

    def _get_replacement_words(self, word):
        candidate_words = []
        for i, char in enumerate(word):
            if not is_chinese(char):
                continue
            repl_chars = getSimilar4Char(char, self.encode_way, self.num_words)
            for repl_char in repl_chars:
                if repl_char == char:
                    continue
                candidate_word = word[:i] + repl_char + word[i+1:]
                candidate_words.append(candidate_word)

        return candidate_words
    
from textattack.transformations import CompositeTransformation
from .Argot_transformation import ChineseShuffleCharacterSwap, ChineseSplittingCharacterSwap, ChineseSynonymWordSwap
from . chinese_word_swap_masked import ChineseWordSwapMaskedLM

mix_ssc = CompositeTransformation([
    ChineseSSCCharacterSwap(),
    ChineseSynonymWordSwap(),
    ChineseWordSwapMaskedLM(),
])


def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True

    return False



