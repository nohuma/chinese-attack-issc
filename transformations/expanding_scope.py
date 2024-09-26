from transformations import ChineseHomophoneCharacterSwap, ChineseWordSwapHowNet, ChineseWordSwapMaskedLM
from textattack.transformations import ChineseMorphonymCharacterSwap, CompositeTransformation


ChineseExpandingScopeWordSwap = CompositeTransformation([
    ChineseWordSwapMaskedLM(),
    ChineseHomophoneCharacterSwap(),
    ChineseMorphonymCharacterSwap(),
    ChineseWordSwapHowNet(),
])