# -*- coding: utf-8 -*-
# 整理清洗数据的脚本
import json
import logging
from collections import Counter
from itertools import combinations, chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, ClusterCentroids, RandomUnderSampler
from scipy.stats import spearmanr, ttest_ind
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils import resample

from CSVReader import CSVReader
from JsonParser import JsonParser

# 设置日志级别为INFO，只记录INFO级别以上的信息
logging.basicConfig(level=logging.INFO)
# 创建FileHandler并配置日志文件名
file_handler = logging.FileHandler('myapp.log')
# 将FileHandler添加到logger中
logger = logging.getLogger()
logger.filter(lambda record: record.levelno == logging.INFO)
logger.addHandler(file_handler)

neg_parser = JsonParser(["C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\negative\\",
                         "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\negative\\"],
                        0,
                        0)  # ,"C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\negative\\"
pos_parser = JsonParser(["C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\positive\\",
                         "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\positive\\"],
                        0,
                        1)  # ,"C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\positive\\"


logging.info('')


def write_as_json():
    for item in neg_parser.data.items():
        print(item[0])
        file_path = prefix_path + negative + item[0]
        # 写入到json
        # 写入到json
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(item[1], json_file, ensure_ascii=False, indent=4)
    for item in pos_parser.data.items():
        print(item[0])
        file_path = prefix_path + positive + item[0]
        # 写入到json
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(item[1], json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # json格式写入到指定文件夹
    prefix_path = "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\final-data-test\\"
    positive = "positive\\"
    negative = "negative\\"
    # 写入
    write_as_json()
    dict = {}
    for item in neg_parser.data.items():
        dict[item[0].split('_')[0]] =1
    for item in pos_parser.data.items():
        dict[item[0].split('_')[0]] =1

    for k in dict.keys():
        print(k) if '@' in k else None



