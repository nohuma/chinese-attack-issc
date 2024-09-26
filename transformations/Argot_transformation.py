# coding=utf-8
import pinyin
import random


import torch
from siamese_cnn.train import BaseCnn

import numpy as np
import pandas as pd

import os
import json

import cv2
from torch.autograd import Variable
from torchvision import transforms

from textattack.transformations import WordSwap, CompositeTransformation


class ChineseSplittingCharacterSwap(WordSwap):
    def __init__(self, struct_list=None):
        if not struct_list:
            struct_list = ['a', 'd']

        f = open("data/chaizi_jt.json", 'r+', encoding='utf-8')
        self.splitting_dict = json.load(f)
        f = open("data/struct_dict_argot.json", 'r+', encoding='utf-8')
        self.struct_dict = json.load(f)

        self.struct_list = struct_list
        self.struct_code_dict = {
            "a": "左右结构",
            "b": "上下结构",
            "c": "独体结构",
            "d": "半包围结构",
            "e": "全包围结构",
            "f": "品字结构",
            "g": "上中下结构",
            "h": "左中右结构"
        }

    def _get_replacement_words(self, word):
        word = list(word)
        candidate_words = set()

        for i in range(len(word)):
            character = word[i]
            if not is_chinese(character):
                continue 
            if character not in self.struct_dict.keys():
                continue
            if self.struct_dict[character] not in self.struct_list:
                continue
            for char_split in self.splitting_dict.keys():
                if character == char_split:
                    for radical in self.splitting_dict[char_split]:
                        temp_word = word[:]
                        temp_word[i] = radical
                        candidate_words.add("".join(temp_word))

        return list(candidate_words)


class ChineseHomophoneCharacterSwap(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by a
    homophone dictionary."""

    def __init__(self):
        homophone_dict = pd.read_csv(
            'data/chinese_homophone_char.txt', header=None, sep="/n", engine='python')
        homophone_dict = homophone_dict[0].str.split("\t", expand=True)

        self.homophone_dict = homophone_dict

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with 1 character
        replaced by a homophone."""
        candidate_words = []
        for i in range(len(word)):
            character = word[i]
            character = pinyin.get(character, format="strip", delimiter=" ")
            if character in self.homophone_dict.values:
                # df is the DataFrame
                for row in range(self.homophone_dict.shape[0]):
                    for col in range(0, 1):
                        if self.homophone_dict._get_value(row, col) == character:
                            for j in range(1, 4):
                                repl_character = self.homophone_dict[col + j][row]
                                if repl_character is None:
                                    break
                                candidate_word = (
                                    word[:i] + repl_character + word[i + 1:]
                                )
                                candidate_words.append(candidate_word)
            else:
                pass
        return candidate_words


class ChineseShuffleCharacterSwap(WordSwap):
    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return word
        else:
            n = len(word)
            i = random.randint(0, n - 2)
            word = list(word)
            temp = word[i]
            word[i] = word[i + 1]
            word[i + 1] = temp
            word = "".join(word)
        return [word]


class ChineseGlyphCharacterSwap(WordSwap):
    def __init__(self, num_words=5):
        with open("data/chaizi-jt.txt", 'r+', encoding='utf-8') as f:
            lines = f.readlines()
        # 拆分字符的字典，key=汉字，value=拆分的部首，形式为list[str,str]
        self.splitting_dict = self.make_dict(lines)
        self.all_chars = list(self.splitting_dict.keys())
        self.all_radicals = self.make_radicals_set(lines)

        self.model = BaseCnn()
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = torch.load('data/checkpoint.pth')['state_dict']

        model_dict_mod = {}
        for key, value in model_dict.items():
            new_key = '.'.join(key.split('.')[1:])
            model_dict_mod[new_key] = value
        self.model.load_state_dict(model_dict_mod)
        self.model.eval()

        self.num_words_left = num_words
        
        # 初次使用将词表中的字转换为图片形式，我们给出了使用的集合在all_chars.zip中。
        # self.initialize_image_data()

    def make_dict(self, lines):
        splitting_dict = dict()

        for line in lines:
            radicals = []
            line = line.strip().replace(" ", "").split('\t')
            splitting_dict[line[0]] = radicals
            radicals.extend(line[1:])

        return splitting_dict

    def make_radicals_set(self, lines):
        all_radicals = set()

        for line in lines:
            line = line.strip().replace(" ", "").split('\t')[1:]
            radicals = list("".join(line))

            all_radicals.update(radicals)
        return all_radicals

    def _get_replacement_words(self, word):
        word = list(word)
        candidate_words = set()
        for i in range(len(word)):
            character = word[i]

            if not is_chinese(character):
                continue
            # 保证拆分的汉字在字典中，保证可以拆分
            if not (character in self.splitting_dict.keys()):
                continue
            char_list = self.glyph(character)
            for char in char_list:
                if char != character:
                    temp_word = word[:]
                    temp_word[i] = char
                    candidate_words.add("".join(temp_word))

        return candidate_words

    def glyph(self, c):
        candidate_chars = set()
        # 将汉字c进行分解，返回的是列表形式
        radicals_list = self.decompose_radicals(c)

        # 替换操作
        # 随机将原始汉字的每个偏旁替换为其他radical
        for i, radical in enumerate(radicals_list):
            # 随机选取一个偏旁集合进行替换
            other = random.choice(list(self.all_radicals))

            if other != radical:
                temp_radicals_list = radicals_list[:]
                temp_radicals_list[i] = other
                # 验证
                char_list = self.val(temp_radicals_list)
                candidate_chars.update(char_list)
        # 删除操作
        # 将原始汉字的每个偏旁删除
        for radical in radicals_list:
            temp_radicals_list = radicals_list[:]
            temp_radicals_list.remove(radical)
            char_list = self.val(temp_radicals_list)

            candidate_chars.update(char_list)

        # 随机选择一个偏旁进行添加
        other = random.choices(list(self.all_radicals), k=200)
        temp_radicals_list = radicals_list[:]
        temp_radicals_list.extend(other)

        char_list = self.val(temp_radicals_list)
        candidate_chars.update(char_list)

        # # 每个都尝试添加一次，耗时比较长
        # for other in self.all_radicals:
        #     # 依次选择一个偏旁
        #     temp_radicals_list = radicals_list[:]
        #     temp_radicals_list.append(other)
        #
        #     char_list = self.val(temp_radicals_list)
        #     candidate_chars.update(char_list)

        candidate_chars_left = self.siamese_similarity(c, candidate_chars)
        return candidate_chars_left

    def val(self, radicals):
        # 检验汉字是否存在与拆分字典中
        char_list = []
        for key in self.splitting_dict:
            char_split_radicals = list("".join(self.splitting_dict[key]))
            # char_split_radicals=['目', '𡕩', '目', '㚇'] key的拆分部首形式
            # 如果当前现有拆分集合是输入集合的子集，就添加到列表中并返回
            if compare(char_split_radicals, radicals[:]):
                char_list.append(key)

        return char_list

    def decompose_radicals(self, character) -> list[str]:
        # use the first split as default
        radicals_list = list(self.splitting_dict[character][0])

        return radicals_list

    def siamese_similarity(self, orig_char, candidate_chars):
        orig_char_emb = torch.unsqueeze(
            self.get_image_emb(orig_char).to('cuda' if torch.cuda.is_available() else 'cpu'), 0)
        candidate_chars_emb = self.get_images_emb(candidate_chars)

        e0 = self.model(Variable(orig_char_emb))
        e = [self.model(Variable(torch.unsqueeze(cand.to('cuda' if torch.cuda.is_available() else 'cpu'), 0))) for cand
             in
             candidate_chars_emb]

        distance_e0_e = torch.tensor(
            [torch.nn.functional.pairwise_distance(e0, ei, 2) for ei in e])

        candidate_chars = np.array(list(candidate_chars))
        _, indices = torch.sort(distance_e0_e)
        better_candidates_chars = candidate_chars[indices[:self.num_words_left]]

        return better_candidates_chars

    def get_image_emb(self, char):
        idx = self.all_chars.index(char)
        path = os.path.join('data/all_chars', f"{idx + 1}.png")
        image_emb = cv2.imread(path)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])

        return transform(image_emb)

    def get_images_emb(self, chars):
        if not chars:
            return []

        images_emb = []
        for char in chars:
            images_emb.append(self.get_image_emb(char))

        return images_emb

    def initialize_image_data(self):
        import pygame
        pygame.init()

        output_dir = os.path.join('data/all_chars')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, char in enumerate(self.all_chars):
            char_path = os.path.join(output_dir, f'{i + 1}.png')
            if os.path.exists(char_path):
                continue

            font = pygame.font.Font("siamese_cnn/simhei.ttf", 100)
            rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
            pygame.image.save(rtext, char_path)

        assert len(os.listdir(path=output_dir)) == len(self.all_chars)


class ChineseSimilarPinyinCharacterSwap(WordSwap):
    def __init__(self):
        homophone_dict_path = 'data/chinese_homophone_char.txt'
        homophone_dict = pd.read_csv(
            homophone_dict_path, header=None, sep="/n", engine='python')
        self.pinyin_dict = homophone_dict[0].str.split("\t", expand=True)
        self.similar_pinyin_dict = {
            "an": "ang",
            "in": "ing",
            "en": "ang",
            'c': 'ch',
            'z': 'zh',
            's': 'sh',
            "ang": "an",
            "ing": "in",
            "eng": "en",
            "ch": 'c',
            "zh": 'z',
            "sh": 's'
        }

    def _get_replacement_words(self, word):
        candidate_words = []
        for i in range(len(word)):
            character = word[i]
            character_pinyin = pinyin.get(
                character, format="strip", delimiter=" ")
            sim_pinyin = []

            for char in self.similar_pinyin_dict.keys():
                if char in character_pinyin:
                    temp_char_pinyin = character_pinyin[:]
                    sim_pinyin.append(temp_char_pinyin.replace(
                        char, self.similar_pinyin_dict[char]))

            sim_pinyin_chars = self.get_words_from_dict(sim_pinyin)
            for repl_character in sim_pinyin_chars:
                candidate_word = (
                    word[:i] + repl_character + word[i + 1:]
                )
                candidate_words.append(candidate_word)

        return candidate_words

    def get_words_from_dict(self, pinyin_list):
        sim_pinyin_chars = []
        for character_pinyin in pinyin_list:
            if character_pinyin in self.pinyin_dict.values:
                for row in range(self.pinyin_dict.shape[0]):
                    for col in range(0, 1):
                        if self.pinyin_dict._get_value(row, col) == character_pinyin:
                            for j in range(1, 4):
                                repl_character = self.pinyin_dict[col + j][row]
                                if repl_character is None:
                                    break

                                sim_pinyin_chars.append(repl_character)
            else:
                pass

        return sim_pinyin_chars


class ChineseSynonymWordSwap(WordSwap):
    def __init__(self, num_words_left=5):
        self.num_words = num_words_left

    def _get_replacement_words(self, word):
        import synonyms
        word_nearby, _ = synonyms.nearby(word, self.num_words)
        return word_nearby


ChineseArgotWordSwap = CompositeTransformation(
    [
        ChineseShuffleCharacterSwap(),
        ChineseSplittingCharacterSwap(),
        ChineseSynonymWordSwap(),
        ChineseHomophoneCharacterSwap(),
        ChineseSimilarPinyinCharacterSwap(),
        ChineseGlyphCharacterSwap(),
    ]
)


def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True

    return False


def compare(s, t):
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False

    return True
