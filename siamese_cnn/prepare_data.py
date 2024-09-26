# -*- coding: utf-8 -*-
import random
import argparse

import pygame
from PIL import Image
import numpy as np
import os
from pypinyin import lazy_pinyin


def set_seed():
    random.seed(args.seed)


def load_chars(path):
    # 获取所有汉字列表
    chars_list = []
    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            chars = [char for char in line.strip()]
            chars_list.append(chars)

    return chars_list


def char2image(chars_list):
    # 通过pygame将汉字转化为黑白图片
    pygame.init()
    output_dir_path = os.path.join('./', args.output_path)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    train_ids_list, test_ids_list = data_split(len(chars_list), args.train_percentage)

    for i, chars in enumerate(chars_list):
        if i in train_ids_list:
            chars_class_path = os.path.join(output_dir_path, 'train', f'character_{i + 1}')
        elif i in test_ids_list:
            chars_class_path = os.path.join(output_dir_path, 'test', f'character_{i + 1}')
        else:
            raise ValueError(f"The length of dataset is out of range, which idx is {i}")

        if not os.path.exists(chars_class_path):
            os.makedirs(chars_class_path)

        for j, char in enumerate(chars):
            char_path = os.path.join(chars_class_path, f'{i + 1}_{j + 1}.png')
            if os.path.exists(char_path):
                continue
            # 文件夹里还有别的类型的字体
            font = pygame.font.Font(r"C://Windows/Fonts/simhei.ttf", 100)
            # 第三个参数为字体颜色，第四个参数为背景颜色。
            rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
            pygame.image.save(rtext, char_path)


def data_split(num_examples, train_percentage):
    set_seed()
    num_train_examples = int(num_examples * train_percentage)
    num_test_examples = num_examples - num_train_examples

    train_dir_path = os.path.join(args.output_path, 'train')
    test_dir_path = os.path.join(args.output_path, 'test')
    if not os.path.exists(train_dir_path):
        os.makedirs(train_dir_path)
    if not os.path.exists(test_dir_path):
        os.makedirs(test_dir_path)

    train_ids_list = random.sample(range(num_examples), num_train_examples)
    test_ids_list = [idx for idx in range(num_examples) if idx not in train_ids_list]

    assert len(test_ids_list) == num_test_examples
    return train_ids_list, test_ids_list


def main():
    path = './形近字.txt'
    chars_list = load_chars(path)
    num_examples = len(chars_list)
    char2image(chars_list)


if __name__ == '__main__':
    # TODO:其他类型的字体
    parser = argparse.ArgumentParser(description='Make the character-pairs dataset')
    parser.add_argument('--output_path', default='characters', type=str,
                        help='Path to store dataset')
    parser.add_argument('--seed', default=718, type=int,
                        help='Fixed seed for Random package')
    parser.add_argument('--dataset_path', default='形近字.txt', type=str,
                        help='Path of raw dataset')
    parser.add_argument('--train_percentage', default=0.7, type=float,
                        help='Percentage of training set')

    global args
    args = parser.parse_args()

    main()
