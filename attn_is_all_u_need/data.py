from collections import Counter

import numpy as np
from config import args

__author__ = 'yscoder@foxmail.com'


class DataLoader:
    def __init__(self, source_path, target_path):
        self.source_words = self.read_data(source_path)
        self.target_words = self.read_data(target_path)

        self.source_word2idx = self.build_index(self.source_words)  # 构建问句的词汇表 such as {"beijing":222, "xx":22  .... }
        self.target_word2idx = self.build_index(self.target_words, is_target=True)  # 构建答句的词汇表

    @staticmethod
    def read_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def build_index(data, is_target=False):
        """
        构建词汇表,忽略出现频率小于设定值的词汇
        :param data:  输入数据集
        :param is_target: True表示是答案句, False表示为问句
        :return:
        """
        chars = [char for line in data.split('\n') for char in line]  # 文档的所有句子中的词汇数组
        chars = [char for char, freq in Counter(chars).items() if freq > args.min_freq]  # 忽略词汇表中词频小于设定值的词汇
        if is_target:  # 如果是答句
            symbols = ['<pad>', '<start>', '<end>', '<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}  # 将特殊的标记添加到问句的词汇表
        else:
            symbols = ['<pad>', '<unk>'] if not args.tied_embedding else ['<pad>', '<start>', '<end>', '<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}  # 将特殊的标记添加到答句的词汇表

    @staticmethod
    def pad(data, word2idx, max_len, is_target=False):
        """
        填充, 如果长度超过设定的最大长度,则做截断操作并添加''<end>'标记,
                               反之,则添加'<end>'标记后用'<pad>'补齐
        :param data:
        :param word2idx:  词汇和id的映射字典
        :param max_len:  设定的语句的最大长度
        :param is_target:
        :return: ndarray
        """
        res = []
        for line in data.split('\n'):
            temp_line = [word2idx.get(char, word2idx['<unk>']) for char in line]
            if len(temp_line) >= max_len:
                if is_target:
                    temp_line = temp_line[:(max_len - 1)] + [word2idx['<end>']]  # padding
                else:
                    temp_line = temp_line[:max_len]  # 截取
            if len(temp_line) < max_len:
                if is_target:
                    temp_line += ([word2idx['<end>']] + [word2idx['<pad>']] * (max_len - len(temp_line) - 1))
                else:
                    temp_line += [word2idx['<pad>']] * (max_len - len(temp_line))
            res.append(temp_line)
        return np.array(res)

    def load(self):
        """
        将问题词汇表和答案词汇表中的数据做补齐,并将词汇完全表示为id的形式
        :return: source_idx

        """
        source_idx = self.pad(self.source_words, self.source_word2idx, args.source_max_len)
        target_idx = self.pad(self.target_words, self.target_word2idx, args.target_max_len, is_target=True)
        return source_idx, target_idx
