# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Params.py
# Time       ：2023/6/7 17:22
# Author     ：Zhang Wenyu
# Description：
"""


class Params():
    def __init__(self, keywords, graph, matrix, node_labels, depth):
        self.depth = depth
        self.keywords = keywords
        self.graph = graph
        self.matrix = matrix
        self.nodes = graph.nodes
        self.node_labels = node_labels

    def set_keywords(self, keywords):
        self.keywords = keywords