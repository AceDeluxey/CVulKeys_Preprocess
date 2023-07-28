# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : metric_cal.py
# Time       ：2023/6/7 16:55
# Author     ：Zhang Wenyu
# Description：
"""
"""
Three different metric:
① 节点出入度
② 节点是否包含漏洞关键词 0和1 
③ 其他行到跟行的距离
(目前实现：其他节点到根行的层数

"""
from Params import Params
import numpy as np
import Utils
import copy
import math


def cal_all_metric(params, metric_dic_list, depth, keywords,d2v_model ):
    temp = copy.deepcopy(metric_dic_list)
    cal_degree(metric_dic_list)
    cal_prob(params, metric_dic_list, keywords,d2v_model )
    cal_distance( metric_dic_list, depth)
    # print(123)



def cal_degree(metric_dic_list):
    """
    # 将每个节点的度归一化
    为了防止最大值太大，使用最大值最小归一化
    :param metric_dic_list:
    :return:
    """
    # 创建一个空数组，用于存储所有字典中的第一列值：
    combined_array = []  # 创建一个空的 (0, 1) 形状的数组
    for metric_dic in metric_dic_list:
        for key, value in metric_dic.items():
            combined_array.append(value[0]) # 将第一列添加到 combined_array 中

        # 写回 并归一化
        for key, value in metric_dic.items():
            value[0] = Utils.sigmoid_normalization(value[0])


def cal_prob(params, metric_dic_list, key_words, d2v_model ):
    """
    判断关键词和每个节点的相似度
    :param params:
    :param metric_dic_list:
    :param key_words:
    :return:

    """
    labels = params.node_labels
    for keyword in key_words:
        key_vector = (d2v_model.infer_vector(doc_words=Utils.tokenize(keyword), steps=20, alpha=0.025))
        for metric_dic in metric_dic_list:
            for key, value in metric_dic.items():
                label = labels[key]
                label_vector = (d2v_model.infer_vector(doc_words=Utils.tokenize(label), steps=20, alpha=0.025))
                sim = Utils.cosine_similarity(key_vector, label_vector)
                value[1] = sim
    return


def cal_prob2(params, metric_dic_list, key_words):
    """
    判断每个节点是否包含关键词,包含为1 不包含为0，2
    :param params:
    :param metric_dic_list:
    :param keywords:
    :return:
    """

    labels = params.node_labels
    for metric_dic in metric_dic_list:
        for key, value in metric_dic.items():
            label = labels[key]
            val = 0.2
            for keyword in key_words:
                if Utils.match_word_node(label, keyword):
                    val = 1
                    break
            value[1] = val
    return


def cal_distance2(metric_dic_list, depth):
    """
    计算每个节点和根节点的距离后，
    转化为度量
    映射成1-1/depth* d*α
    α 是系数 控制缩放，保证最小值不是0
    :param metric_dic_list:
    :return:
    """
    α = 0.85
    unit = 1/depth
    for metric_dic in metric_dic_list:
        for key, value in metric_dic.items():
            value[2] = 1-value[2]*unit*α
    return


def cal_distance(metric_dic_list, depth):
    """
    计算每个节点和根节点的距离后，
    转化为度量
    映射成1乘以0.8的k次方
    α 是系数 控制缩放，保证最小值不是0
    :param metric_dic_list:
    :return:
    """
    α = 0.8
    # unit = 1/depth
    for metric_dic in metric_dic_list:
        for key, value in metric_dic.items():
            value[2] = 1*math.pow(α,value[2])
    return