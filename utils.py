# coding: UTF-8
import os
import yaml
import time
import h5py
import pickle
from datetime import timedelta
import random
import string
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.safe_load(open(path, 'r', encoding='utf-8')))


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_data(cache_file_h5py, cache_file_pickle):
    """
    load data from h5py and pickle cache files
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):

        raise RuntimeError("############################ERROR##############################\n. "
                           "请先准备数据集")
    f_data = h5py.File(cache_file_h5py, 'r')
    # return narray
    # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
    input_ids = f_data['input_ids'][()]
    attention_mask = f_data['attention_mask'][()]
    label_list = f_data['label_list'][()]

    label2index = None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        label2index = pickle.load(data_f_pickle)
    return label2index,input_ids,attention_mask,label_list

def load_labels(cache_file_h5py):
    """
    load data from h5py and pickle cache files
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py):

        raise RuntimeError("############################ERROR##############################\n. "
                           "请先准备数据集")
    f_data = h5py.File(cache_file_h5py, 'r')
    # return narray
    # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
    input_ids = f_data['input_ids'][()]
    attention_mask = f_data['attention_mask'][()]

    return input_ids,attention_mask

def load_formula(cache_file_h5py, cache_file_pickle):
    """
    load data from h5py and pickle cache files
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):

        raise RuntimeError("############################ERROR##############################\n. "
                           "请先准备数据集")
    f_data = h5py.File(cache_file_h5py, 'r')
    # return narray
    # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
    formula = f_data['X_mathml'][()]

    word2index, label2index = None, None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index = pickle.load(data_f_pickle)
    return word2index, label2index, formula


def load_embed_data(embedding_pickle):
    '''加载预训练向量 narray'''
    embeddings = None
    with open(embedding_pickle, 'rb') as data_f_pickle:
        embeddings = pickle.load(data_f_pickle)
    return embeddings


def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def build_label_relation_matrix(y_train, num_classes, threshold=0.1, symmetric=False):
    """
    构建标签关系矩阵
    
    Args:
        y_train: 训练集标签，形状为 [n_samples, n_labels]，二进制矩阵
        num_classes: 标签类别数量
        threshold: 共现概率阈值，低于此值的边将被移除
        symmetric: 是否对称化处理
        
    Returns:
        labels_relation: 标签关系矩阵，形状为 [n_labels, n_labels]
    """
    # 初始化计数矩阵
    cooccurrence = np.zeros((num_classes, num_classes))
    label_counts = np.sum(y_train, axis=0)  # 每个标签出现的次数
    
    # 计算共现次数
    for sample in y_train:
        present_labels = np.where(sample == 1)[0]
        for i in present_labels:
            for j in present_labels:
                if i != j:
                    cooccurrence[i, j] += 1
    
     # 计算条件概率 P(c|p) = count(p,c)/count(p)
    label_counts = np.maximum(label_counts, 1)
    confidence = np.zeros((num_classes, num_classes))
    
    for p in range(num_classes):
        for c in range(num_classes):
            if p != c:
                # 计算 confidence(p,c)
                conf_p_c = cooccurrence[p, c] / label_counts[p]
                
                # 正确的映射：labels_relation[c,p] = confidence(p,c)
                # 这样标签 p 的信息会流向标签 c
                confidence[c, p] = conf_p_c
            else:
                confidence[c, p] = 1.0
    
    # 应用阈值
    if threshold > 0:
        confidence = np.where(confidence > threshold, confidence, 0)
    
    # 对称化处理（如果需要）
    if symmetric:
        confidence = np.maximum(confidence, confidence.T)
    
    return confidence


# 标签图可视化，后续可能使用
def visualize_label_relation(labels_relation, label_names=None, top_n=None, 
                            filename='label_relation_matrix.png', show_values=False):
    """
    可视化标签关系矩阵
    
    Args:
        labels_relation: 标签关系矩阵，形状为 [n_labels, n_labels]
        label_names: 标签名称列表，如果为None则使用索引
        top_n: 只显示关系最强的前N个标签，如果为None则显示所有
        filename: 保存的文件名
        show_values: 是否在热力图中显示具体数值
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # 如果没有提供标签名称，使用索引
    if label_names is None:
        label_names = [str(i) for i in range(labels_relation.shape[0])]
    
    # 如果指定了top_n，选择关系最强的标签
    if top_n is not None and top_n < len(label_names):
        # 计算每个标签的总关系强度
        total_strength = np.sum(labels_relation, axis=1) + np.sum(labels_relation, axis=0)
        # 选择关系最强的top_n个标签
        top_indices = np.argsort(total_strength)[-top_n:]
        labels_relation = labels_relation[top_indices][:, top_indices]
        label_names = [label_names[i] for i in top_indices]
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图
    annot = labels_relation if show_values else False
    sns.heatmap(labels_relation, annot=annot, fmt='.2f', cmap='YlGnBu', 
               xticklabels=label_names, yticklabels=label_names)
    
    # 设置标题和标签
    plt.title('Label Relation Matrix')
    plt.xlabel('From Label')
    plt.ylabel('To Label')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {filename}")