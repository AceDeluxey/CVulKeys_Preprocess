# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : test.py
# Time       ：2023/5/26 14:46
# Author     ：Zhang Wenyu
# Description：
"""


import graphviz
import numpy as np
import networkx as nx
from collections import deque
from Params import  Params
# from networkx.drawing.nx_agraph import read_dot
from Utils import  match_word_node
import networkx as nx
import pydot


def get_keywords(file_path):
    # list = ["socket", "minus","nmap"]
    word_list = []
    with open(file_path, 'r') as file:
        for line in file:
            word = line.strip()  # Remove any leading/trailing whitespace or newline characters
            word_list.append(word)
    return word_list


def find_root_line(params):
    """
    找到每个关键词对应的root line，然后生成按照每个关键词的度大小排序，排在第一个的是度最大的
    :param params:
    :return:
    """
    key_words = params.keywords
    nodes = params.nodes
    labels = params.node_labels
    graph = params.graph

    # 包含keyword的节点
    dic ={}
    for keyword in key_words:
        # list记录每个关键词对应的节点id的list
        list = []
        for node in nodes:
            label = labels[node]
            if match_word_node(label,keyword):
                list.append(node)
        dic[keyword] = list

    """
        存储 {'value': {'minus': '1000105'}, 'weight': 2} 的list
    """
    index_list = []

    # 遍历字典的键值对,求每个关键字对应最大度的行
    for key, value in dic.items():
        degree = -1
        index = -1
        if len(value) !=0:
            for i in value:
                temp = graph.degree(i)
                if temp>degree:
                    index = i
                    degree = temp
        else:
            continue
        temp_dic = {}
        temp_dic["value"] = {key:index}
        temp_dic["weight"] = degree
        index_list.append(temp_dic)
    # print(123)
    # 按照最大度权重进行排序
    index_list = sorted(index_list, key=lambda x: x["weight"], reverse=True)
    return index_list


def slice_one(graph,start_node,target_depth):
    """
    针对每个graph，每个root_Lien切一次
    :param graph: graph对象
    :param start_node:  根行index
    :param target_depth: 切割深度
    :param metric_dic: 每个节点的三个指标
    :return:
    """
    # 创建每个节点的度量向量
    metric_dic = {}
    # 起始节点的第三个指标
    temp_list = np.zeros(3)
    temp_list[0] = graph.degree(start_node)
    temp_list[2] = 0
    metric_dic[start_node] = temp_list

    # 创建一个空的有向图用于存储子图
    subgraph = nx.MultiDiGraph()

    # 使用双向队列作为 BFS 的辅助数据结构
    queue = deque([(start_node, 0)])  # 存储节点和深度的元组

    # 从原始的 graph 获取节点和边的标签属性
    node_labels = nx.get_node_attributes(graph, 'label')
    edge_labels = nx.get_edge_attributes(graph, 'label')

    visited = set()  # 用于记录已经访问过的节点
    added_edges = set()  # 用于记录已经添加过的边
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        # 将当前节点添加到子图中
        subgraph.add_node(node)

        # 如果深度小于等于depth，则继续搜索下一层相邻节点
        if depth < target_depth:
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    # 两指标
                    temp_list = np.zeros(3)
                    temp_list[0] = graph.degree(neighbor)
                    temp_list[2] = depth+1
                    metric_dic[neighbor] = temp_list

                # 获取从 Node1 到 Node2 的所有边的属性
                edges_data = graph.get_edge_data(node, neighbor)
                # 遍历所有边的属性
                for key, data in edges_data.items():
                    edge_label = data['label']
                    edge = (node, neighbor, edge_label)
                    if edge in added_edges:
                        continue  # 如果该边已经添加过，则跳过
                    added_edges.add(edge)
                    subgraph.add_edge(node, neighbor, label=edge_label)

                queue.append((neighbor, depth + 1))

    # 将节点标签属性存储为 DOT 文件中的 label 属性
    nx.set_node_attributes(subgraph, node_labels, 'label')
    # 将边标签属性存储为 DOT 文件中的 label 属性
    nx.set_edge_attributes(subgraph, edge_labels, 'label')

    return subgraph, metric_dic


def slice(params, index_list, k):
    """
    针对每个graph，切多个关键词的子图
    :param params: 图的信息
    :param k: 生成前K个子图
    :param index_dic: 每个关键词度最大的node id
    :return: list(切的子图列表),metric_dic_list（每个图节点的度量列表）
    """
    depth = params.depth
    list = []
    metric_dic_list=[]
    for i in range(min(k, len(index_list))):
        index_dic = index_list[i]
        "index_dic  {'value': {'minus': '1000105'}, 'weight': 2} "
        index_dic = index_dic['value']
        for key, value in index_dic.items():
            graph, metric_dic = slice_one(params.graph, value, depth)
            list.append(graph)
            metric_dic_list.append(metric_dic)
    return list, metric_dic_list


def convert_to_multigraph(subgraph):
    # 创建一个空的 MultiGraph
    multi_graph = nx.MultiGraph()

    # 获取子图的标签
    label = subgraph.get_attributes().get('label', '')

    # 添加节点和边
    for node in subgraph.get_node_list():
        node_name = node.get_name()
        node_label = node.get_attributes().get('label', '')
        multi_graph.add_node(node_name, label=node_label)

    for edge in subgraph.get_edge_list():
        src = edge.get_source()
        dst = edge.get_destination()
        edge_label = edge.get_attributes().get('label', '')  # 获取边的标签
        multi_graph.add_edge(src, dst, label=edge_label)

    # 设置 MultiGraph 的标签
    multi_graph.graph["label"] = label

    return multi_graph


def read_dot(file_name):
    # graph  = pydot.graph_from_dot_file(file_name)
    graph = nx.drawing.nx_pydot.read_dot(file_name)
    # 有子图
    if graph.number_of_nodes()==0:
        graph = pydot.graph_from_dot_file(file_name)[0]

        # 获取子图0
        subgraphs = graph.get_subgraph_list()
        if len(subgraphs) != 0:
            subgraph = subgraphs[0]
            graph = convert_to_multigraph(subgraph)
            # 转换为有向图
            directed_graph = graph.to_directed()
            graph = directed_graph
        else:
            graph = nx.MultiGraph()
    return graph


def create_matrix(directed_graph):
    try:

        adjacency_matrix = nx.to_numpy_matrix(directed_graph)
    except Exception as e:
        # 处理异常的代码
        adjacency_matrix = None
        print("发生了一个异常:", e)
    return adjacency_matrix


def get_node_labels(graph):
    # 获取所有节点
    nodes = graph.nodes()
    # 获取节点的label属性字典
    labels = nx.get_node_attributes(graph, 'label')
    return labels


def get_line(graph,params):
    """
    Get every line of source dot file
        to calculate globlal vector
    :param graph:
    :return: line_list,weight_list
    """
    line_list =[]
    weight_list = []
    # 获取节点列表并排序
    nodes = sorted(graph.nodes())
    nodes_len = len(nodes)
    labels = params.node_labels
    # 遍历节点
    for node in nodes:
        node_label = labels[node]
        line_list.append(node_label)
        node_degree = graph.degree[node]
        # print("Node:", node)
        # print("Label:", node_label)
        # print("Degree:", node_degree)
        # line_list.append(node_label)
        weight_list.append(node_degree/(2*nodes_len))
    return line_list, weight_list



def slice_for_one_dot(file_name, key_words,depth,k):
    """
    读取dot文件，为dot文件创建多个子图
    :param file_name:  文件名
    :param key_words: 关键词
    :param depth: 搜索深度
    :return:
    """
    graph = read_dot(file_name)



    # 获取邻接矩阵
    matrix = create_matrix(graph)

    # 获取labels
    node_labels = get_node_labels(graph)

    params = Params(key_words, graph, matrix, node_labels,depth)

    # Get lines from original file
    line_list, weight_list = get_line(graph,params)

    # 得到每个关键词度最大的node id
    index_list = find_root_line(params)

    # 创建度量字典list
    metric_dic_list = []
    # 切片
    slice_graphs, metric_dic_list = slice(params, index_list,k)

    # print(len(line_list))
    # print(len(weight_list))
    return slice_graphs, metric_dic_list, line_list, weight_list, params
    # # 将子图保存为 DOT 文件
    # nx.drawing.nx_pydot.write_dot(slice_graphs[0], 'subgraph.dot')


if __name__ == '__main__':
    key_words = get_keywords()
    file_name = "Data/CVE_raw_000062516_CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_bad.dot"
    slice_for_one_dot(file_name, key_words,3)

    print(123)


