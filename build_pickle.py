# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : build_pickle.py
# Time       ：2023/7/8 18:41
# Author     ：Zhang Wenyu
# Description：
"""
# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : getData.py
# Time       ：2023/7/4 22:57
# Author     ：Zhang Wenyu
# Description：
"""
import pickle
import os
import networkx as nx


def read_data_build_pickle(folder_path):
    # 存储所有数据的列表
    all_data = []
    # 存储文本、标签和权重的列表
    global_texts = []
    labels = []
    global_weights = []
    sub_list = []
    metric_list = []
    num_class = 0
    label = 0
    # 记录label的序号
    label_name = []
    # 遍历文件夹
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)

        # 判断是否为文件夹
        if os.path.isdir(folder_dir):
            folder_data = []
            label_name.append(folder_name)

            folder_paths = []
            # 找到一个类别文件夹的所有list_weights.pickel
            for root, dirs, files in os.walk(folder_dir):
                if 'line_weight_lists.pkl' in files:
                    folder_paths.append(root)
            # 遍历数据文件夹
            for data_folder_dir in folder_paths:
                # data_folder_dir = os.path.join(folder_dir, data_folder_name)
                flag = True
                # 判断是否为文件夹
                if os.path.isdir(data_folder_dir):
                    print(data_folder_dir)
                    # 遍历pickle文件
                    sub_lines = []
                    sub_metrics = []
                    line = []
                    weight = []
                    for file_name in os.listdir(data_folder_dir):
                        file_path = os.path.join(data_folder_dir, file_name)

                        # 判断是否为pickle文件
                        if file_name.endswith("line_weight_lists.pkl"):
                            # print(file_path)
                            # 读取pickle文件
                            with open(file_path, "rb") as f:
                                data = pickle.load(f)
                                # 提取文本列表、标签和权重
                                # 解压缩zip对象
                                try:
                                    line, weight = zip(*data)
                                except Exception as e:
                                    # 发生任意异常时跳过当前循环

                                    print(e)
                                    flag = False
                                    break
                                # = zip_data

                        if "subgraph0" in file_name:
                            # 读取subgraph文件
                            sub_lines = get_lines(file_path)

                        if "metric" in file_name:
                            """
                            读取这个列表的第一个，也就是第一个subgraph的
                            """
                            with open(file_path, "rb") as f:
                                metrics = pickle.load(f)

                                if len(metrics) == 0:
                                    sub_metrics = []
                                else:
                                    metric = metrics[0]
                                    sorted_keys = sorted(metric.keys())
                                    sub_metrics = [metric[key] for key in sorted_keys]
                                # 排序dic，按照node的标号顺序从上到下排成list

                    if flag:
                        if len(line) > 0:
                            # 将文本、标签和权重添加到对应的列表中
                            global_texts.append(list(line))
                            labels.append(label)
                            global_weights.append(list(weight))
                            sub_list.append(sub_lines)
                            metric_list.append(sub_metrics)
                        # print("sub_lines: {}".format(len(sub_lines)))
                        # print("sub_metrics: {}".format(len(sub_metrics)))
        label = label + 1
    # 这里返回的label就是有几类

    pickle_name = os.path.join(folder_path, 'all_data.pickle')
    data = [global_texts, global_weights, metric_list, sub_list, labels, label_name]
    with open(pickle_name, 'wb') as file:
        pickle.dump(data, file,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return global_texts, global_weights, metric_list, sub_list, labels, label


def get_lines(file_name):
    """
    Get every line of source dot file
    :param graph:
    :return: line_list,weight_list
    """
    graph = nx.drawing.nx_pydot.read_dot(file_name)
    line_list = []
    weight_list = []
    # 获取节点列表并排序
    nodes = sorted(graph.nodes())
    # 节点可能有'\\n' 不知道为什么,去掉不合法的
    for node in nodes:
        if not check_numeric_string(node):
            nodes.remove(node)
    nodes_len = len(nodes)
    labels = nx.get_node_attributes(graph, 'label')
    # 遍历节点
    for node in nodes:
        node_label = labels[node]
        line_list.append(node_label)

    return line_list


def check_numeric_string(input_string):
    """

    :param input_string:
    :return:
    """
    try:
        float(input_string)  # 尝试将字符串转换为浮点数
        return True  # 转换成功，字符串只包含数字
    except ValueError:
        return False  # 转换失败，字符串包含非数字字符


def convert_lines_to_vectors():
    return


if __name__ == '__main__':
    folder_path = "../NewVul_Model\data\\small"
    read_data_build_pickle(folder_path)
    print('over')