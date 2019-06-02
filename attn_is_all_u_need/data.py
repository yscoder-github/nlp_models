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
        build word mapping table 
        1. Ignores words that are less than the specified word frequency (args.min_freq)
        2. Adds the special symbol in the sentence to the front of the mapping table
        :param data:  input dataset 
        :param is_target: True: target sentece, False: source sentence 
        :return: word mapping table 
        """
        chars = [char for line in data.split('\n') for char in line] 
        chars = [char for char, freq in Counter(chars).items() if freq > args.min_freq] 
        if is_target:
            symbols = ['<pad>', '<start>', '<end>', '<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}  
        else:
            symbols = ['<pad>', '<unk>'] if not args.tied_embedding else ['<pad>', '<start>', '<end>', '<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}

    @staticmethod
    def pad(data, word2idx, max_len, is_target=False):
        """
        Preprocessing the  sentence to setting length 
        :param data: input sentence list 
        :param word2idx:  word2idx mapping table 
        :param max_len:  setting max sentence length 
        :return: ndarray
        """
        res = []
        for line in data.split('\n'):
            temp_line = [word2idx.get(char, word2idx['<unk>']) for char in line]  
            if len(temp_line) >= max_len: # do truncatation 
                if is_target:
                    temp_line = temp_line[:(max_len - 1)] + [word2idx['<end>']] 
                else:
                    temp_line = temp_line[:max_len]  
            if len(temp_line) < max_len: # do padding 
                if is_target:
                    temp_line += ([word2idx['<end>']] + [word2idx['<pad>']] * (max_len - len(temp_line) - 1))
                else:
                    temp_line += [word2idx['<pad>']] * (max_len - len(temp_line))
            res.append(temp_line)
        return np.array(res)

    def load(self):
        """
        Preprocessing the training data 
        :return: source_idx, target_idx 

        """
        source_idx = self.pad(self.source_words, self.source_word2idx, args.source_max_len)
        target_idx = self.pad(self.target_words, self.target_word2idx, args.target_max_len, is_target=True)
        return source_idx, target_idx
