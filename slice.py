# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : slice.py
# Time       ：2023/5/27 16:14
# Author     ：Zhang Wenyu
# Description：
"""
import os
import networkx as nx
import find_root_line as find
import metric_cal as mc
import pickle
from gensim.models.doc2vec import Doc2Vec

from Params import  Params
def get_dot_files(folder_path):
    dot_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dot"):
                dot_files.append(os.path.join(root, file))
    return dot_files

# 生成子图的代码，根据您之前提供的逻辑实现

# 定义文件夹路径
folder = "../NewVul_Model\data\\tian"
# folder = "Data"
# depth of BFS
depth = 3

model_name = "model/d2v.bin"
d2v_model = Doc2Vec.load(model_name)
# d2v_model = None

# 获取文件夹中的所有 .dot 文件
dot_files = get_dot_files(folder)

word_list = find.get_keywords('keyword_list.txt')
for dot_file in dot_files:
    # 构建完整的文件路径
    dot_file_path = dot_file
    print(dot_file_path)

    # 判断有没有处理过了
    folder_path = os.path.dirname(dot_file_path) # dot的文件夹
    file_name = os.path.basename(dot_file_path) # dot的名字
    file_name = file_name.split(".dot")[0] # dot的文件名
    target_folder = os.path.join(folder_path,  file_name)
    if os.path.isdir(target_folder):
        file_path = os.path.join(target_folder, "line_weight_lists.pkl")
        if os.path.isfile(file_path):
            print("skip")
            continue


    if "subgraph" in dot_file_path:
        continue
    # 读取 .dot 文件中的图
    # 切的子图，度量值列表，params,取最大的1个子图
    slice_graphs, metric_dic_lists,line_list, weight_list, params = find.slice_for_one_dot(dot_file_path, word_list,depth,1)
    line_weight_lists = list(zip(line_list, weight_list))
    # processing metric_dic_list
    mc.cal_all_metric(params, metric_dic_lists, depth, word_list, d2v_model )
    #

    # 创建对应的文件夹
    output_folder = os.path.splitext(dot_file)[0]  # 使用 .dot 文件名作为文件夹名
    # output_folder_path = os.path.join(folder, output_folder)
    output_folder_path = output_folder
    os.makedirs(output_folder_path, exist_ok=True)

    # 构建子图文件的路径
    index=0
    for subgraph in slice_graphs:
        file_name = "subgraph"+str(index)+".dot"
        subgraph_file_path = os.path.join(output_folder_path,file_name )
        index+=1
        # 将子图保存为 .dot 文件
        nx.drawing.nx_pydot.write_dot(subgraph, subgraph_file_path)
        # 存子图的metrics

    # 获取当前工作目录
    current_dir = os.getcwd()

    """
    metric_dic_lists.pkl is the weights of subgraphs 
    """
    if len(metric_dic_lists) != 0:
        file_name = "metric_dic_lists.pkl"
        file_name = os.path.join(output_folder_path, file_name)
        with open(file_name, 'wb') as f:
            pickle.dump( metric_dic_lists, f)

    """
        line_weight_lists.pkl is the lines of original dot and weights
    """

    if len(line_list) !=0:
        file_name = "line_weight_lists.pkl"
        file_name = os.path.join(output_folder_path, file_name)

        with open(file_name, 'wb') as f:
            pickle.dump( line_weight_lists, f)
