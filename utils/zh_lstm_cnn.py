import numpy as np
import re
import jieba

import torch.nn as nn
import torch
from torch.nn import functional as F

import os
import json

from textattack.models.helpers import EmbeddingLayer
from textattack.model_args import TEXTATTACK_MODELS
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.shared import utils


class ZHGloveEmbeddingLayer(EmbeddingLayer):
    EMBEDDING_PATH = "/root/autodl-tmp/word_embedding"

    def __init__(self, emb_layer_trainable=True):
        glove_path = ZHGloveEmbeddingLayer.EMBEDDING_PATH
        glove_word_list_path = os.path.join(glove_path, "ZHglove.wordlist.npy")
        word_list = np.load(glove_word_list_path)
        glove_matrix_path = os.path.join(glove_path, "ZHglove.300d.mat.npy")
        embedding_matrix = np.load(glove_matrix_path)
        super().__init__(embedding_matrix=embedding_matrix, word_list=word_list)
        self.embedding.weight.requires_grad = emb_layer_trainable


def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    try:
        if re.search("[\u4e00-\u9FFF]", s):
            seg_list = jieba.cut(s, cut_all=False)
            s = " ".join(seg_list)
        else:
            s = " ".join(s.split())
    except Exception:
        s = " ".join(s.split())

    homos = """˗৭Ȣ𝟕бƼᏎƷᒿlO`ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"""
    exceptions = """'-_*@"""
    filter_pattern = homos + """'\\-_\\*@"""
    # TODO: consider whether one should add "." to `exceptions` (and "\." to `filter_pattern`)
    # example "My email address is xxx@yyy.com"
    filter_pattern = f"[\\w{filter_pattern}]+"
    words = []
    for word in s.split():
        # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the word.
        word = word.lstrip(exceptions)
        filt = [w.lstrip(exceptions) for w in re.findall(filter_pattern, word)]
        words.extend(filt)
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words


class ZHGloveTokenizer:
    def __init__(self, word_id_map={}, pad_token_id=None, unk_token_id=None, max_length=128):
        self.word2id = word_id_map
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

    def _process_text(self, text_input):
        """A text input may be a single-input tuple (text,) or multi-input
        tuple (text, text, ...).

        In the single-input case, unroll the tuple. In the multi-input
        case, raise an error.
        """
        if isinstance(text_input, tuple):
            if len(text_input) > 1:
                raise ValueError(
                    "Cannot use `GloveTokenizer` to encode multiple inputs"
                )
            text_input = text_input[0]
        text_input = self.enable_truncation(text_input)
        return text_input

    def _split_text(self, text):
        text = self._process_text(text)
        words = words_from_text(text)
        return words

    def encode(self, text):
        # words
        tokens = self._split_text(text)

        output = []
        for token in tokens:
            if token in self.word2id:
                output.append(self.word2id[token])
            else:
                output.append(self.unk_token_id)
        output = self.enable_truncation(output)
        output = self.enable_padding(output)

        return output

    def batch_encode(self, input_text_list):
        """The batch equivalent of ``encode``."""
        input_text_list = list(map(self._process_text, input_text_list))

        outputs = []
        for input_text in input_text_list:
            outputs.append(self.encode(input_text))

        # if not isinstance(outputs, torch.Tensor):
        #     outputs = torch.tensor(outputs)
        return outputs

    def __call__(self, input_texts):
        if isinstance(input_texts, list):
            return self.batch_encode(input_texts)
        else:
            return self.encode(input_texts)

    def enable_truncation(self, ids):
        return ids[:self.max_length]

    def enable_padding(self, ids):
        while (len(ids) < self.max_length):
            ids.append(self.pad_token_id)
        assert len(ids) == self.max_length

        return ids


class ZHLSTMForClassification(nn.Module):
    """A long short-term memory neural network for text classification.

    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        hidden_size=150,
        depth=1,
        dropout=0.3,
        num_labels=2,
        max_seq_length=128,
        model_path=None,
        emb_layer_trainable=True,
    ):
        super().__init__()
        self._config = {
            "architectures": "LSTMForClassification",
            "hidden_size": hidden_size,
            "depth": depth,
            "dropout": dropout,
            "num_labels": num_labels,
            "max_seq_length": max_seq_length,
            "model_path": model_path,
            "emb_layer_trainable": emb_layer_trainable,
        }
        if depth <= 1:
            # Fix error where we ask for non-zero dropout with only 1 layer.
            # nn.module.RNN won't add dropout for the last recurrent layer,
            # so if that's all we have, this will display a warning.
            dropout = 0
        self.drop = nn.Dropout(dropout)
        self.emb_layer_trainable = emb_layer_trainable
        self.emb_layer = ZHGloveEmbeddingLayer(
            emb_layer_trainable=emb_layer_trainable)
        self.word2id = self.emb_layer.word2id
        self.encoder = nn.LSTM(
            input_size=self.emb_layer.n_d,
            hidden_size=hidden_size // 2,
            num_layers=depth,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = ZHGloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

        if model_path is not None:
            self.load_from_disk(model_path)
        self.eval()

    def load_from_disk(self, model_path):
        # TODO: Consider removing this in the future as well as loading via `model_path` in `__init__`.
        import warnings

        warnings.warn(
            "`load_from_disk` method is deprecated. Please save and load using `save_pretrained` and `from_pretrained` methods.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.load_state_dict(load_cached_state_dict(model_path))
        self.eval()

    def save_pretrained(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(
            state_dict,
            os.path.join(output_path, "pytorch_model.bin"),
        )
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self._config, f)

    @classmethod
    def from_pretrained(cls, name_or_path):
        """Load trained LSTM model by name or from path.

        Args:
            name_or_path (:obj:`str`): Name of the model (e.g. "lstm-imdb") or model saved via :meth:`save_pretrained`.
        Returns:
            :class:`~textattack.models.helpers.LSTMForClassification` model
        """
        if name_or_path in TEXTATTACK_MODELS:
            # path = utils.download_if_needed(TEXTATTACK_MODELS[name_or_path])
            path = utils.download_from_s3(TEXTATTACK_MODELS[name_or_path])
        else:
            path = name_or_path

        config_path = os.path.join(path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "architectures": "LSTMForClassification",
                "hidden_size": 150,
                "depth": 1,
                "dropout": 0.3,
                "num_labels": 2,
                "max_seq_length": 128,
                "model_path": None,
                "emb_layer_trainable": True,
            }
        del config["architectures"]
        model = cls(**config)
        state_dict = load_cached_state_dict(path)
        model.load_state_dict(state_dict)
        return model

    def forward(self, _input):
        # ensure RNN module weights are part of single contiguous chunk of memory
        self.encoder.flatten_parameters()

        emb = self.emb_layer(_input.t())
        emb = self.drop(emb)

        output, hidden = self.encoder(emb)

        output = torch.max(output, dim=0)[0]

        output = self.drop(output)
        pred = self.out(output)
        return pred

    def get_input_embeddings(self):
        return self.emb_layer.embedding


class ZHWordCNNForClassification(nn.Module):
    """A convolutional neural network for text classification.

    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        hidden_size=150,
        dropout=0.3,
        num_labels=2,
        max_seq_length=128,
        model_path=None,
        emb_layer_trainable=True,
    ):
        super().__init__()
        self._config = {
            "architectures": "WordCNNForClassification",
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_labels": num_labels,
            "max_seq_length": max_seq_length,
            "model_path": model_path,
            "emb_layer_trainable": emb_layer_trainable,
        }
        self.drop = nn.Dropout(dropout)
        self.emb_layer = ZHGloveEmbeddingLayer(
            emb_layer_trainable=emb_layer_trainable)
        self.word2id = self.emb_layer.word2id
        self.encoder = CNNTextLayer(
            self.emb_layer.n_d, widths=[3, 4, 5], filters=hidden_size
        )
        d_out = 3 * hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = ZHGloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

        if model_path is not None:
            self.load_from_disk(model_path)
        self.eval()

    def load_from_disk(self, model_path):
        # TODO: Consider removing this in the future as well as loading via `model_path` in `__init__`.
        import warnings

        warnings.warn(
            "`load_from_disk` method is deprecated. Please save and load using `save_pretrained` and `from_pretrained` methods.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.load_state_dict(load_cached_state_dict(model_path))
        self.eval()

    def save_pretrained(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self._config, f)

    @classmethod
    def from_pretrained(cls, name_or_path):
        """Load trained Word CNN model by name or from path.

        Args:
            name_or_path (:obj:`str`): Name of the model (e.g. "cnn-imdb") or model saved via :meth:`save_pretrained`.
        Returns:
            :class:`~textattack.models.helpers.WordCNNForClassification` model
        """
        if name_or_path in TEXTATTACK_MODELS:
            path = utils.download_from_s3(TEXTATTACK_MODELS[name_or_path])
        else:
            path = name_or_path

        config_path = os.path.join(path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "architectures": "WordCNNForClassification",
                "hidden_size": 150,
                "dropout": 0.3,
                "num_labels": 2,
                "max_seq_length": 128,
                "model_path": None,
                "emb_layer_trainable": True,
            }
        del config["architectures"]
        model = cls(**config)
        state_dict = load_cached_state_dict(path)
        model.load_state_dict(state_dict)
        return model

    def forward(self, _input):
        emb = self.emb_layer(_input)
        emb = self.drop(emb)

        output = self.encoder(emb)

        output = self.drop(output)
        pred = self.out(output)
        return pred

    def get_input_embeddings(self):
        return self.emb_layer.embedding


class CNNTextLayer(nn.Module):
    def __init__(self, n_in, widths=[3, 4, 5], filters=100):
        super().__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, Ci, len, d)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs1
        ]  # [(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]
        x = torch.cat(x, 1)
        return x
