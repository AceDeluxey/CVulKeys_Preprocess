# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Utils.py
# Time       ：2023/6/7 19:08
# Author     ：Zhang Wenyu
# Description：
"""
import  numpy as np
import  re
from nltk.stem import WordNetLemmatizer


import numpy as np


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = abs(dot_product) / (norm1 * norm2)
    return similarity


def match_word_node(s,word):
    """
    现在请你写一个判断词语word是否在字符串s里的方法，匹配到的条件有，大写的word在s里，小写的word在s里，word去除特殊符号后在字符串s内
    :param s:
    :param word:
    :return:
    """
    # 匹配节点是否包含关键字
    uppercase_word = word.upper()
    lowercase_word = word.lower()
    clean_word = re.sub(r'\W+', '', word)
    if word in s:
        return True
    if uppercase_word in s:
        return True
    if lowercase_word in s:
        return True
    if clean_word in s:
        return True

    return False


def max_min_normalization(data, value):
    """
    最大最小归一化
    :param data:
    :param value:
    :return:
    """
    min_value = min(data)  # 计算均值
    max_value = max(data)  # 计算标准差
    return (value - min_value) / (max_value - min_value)


def z_score_normalization(data, value):
    """
    Z-Score 标准化
    :param data:
    :param value:
    :return:
    """
    mean = np.mean(data)  # 计算均值
    std = np.std(data)  # 计算标准差

    normalized_value = (value - mean) / std  # Z-Score 标准化

    return value


def sigmoid_normalization(data):
    """
    sigmoid_normalization
    :param data:
    :return:
    """
    normalized_data = 1 / (1 + np.exp(-data))
    return normalized_data


def tokenize(sentence):
    """
    对sentence分词
    :param sentence:
    :return:
    """
    text = re.sub('[^a-zA-Z]', ' ', sentence)
    words = text.split()
    # 解析驼峰
    new_words = []
    for word in words:
        new_words.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', word))
    lowercase_words = [word.lower() for word in new_words]
    lemmatizer = WordNetLemmatizer()
    lem_words = [lemmatizer.lemmatize(w, pos='n') for w in lowercase_words]
    # stopwords = {}.fromkeys([line.rstrip() for line in open('utils/stopwords.txt')])
    # eng_stopwords = set(stopwords)
    # words = [w for w in lem_words if w not in eng_stopwords]
    words = lem_words
    return words