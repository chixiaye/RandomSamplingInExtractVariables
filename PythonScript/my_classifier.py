from collections import Counter

import graphviz as graphviz
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text

from JsonParser import JsonParser
import logging
import pandas as pd


# 设置日志级别为INFO，只记录INFO级别以上的信息
logging.basicConfig(level=logging.INFO)
# 创建FileHandler并配置日志文件名
file_handler = logging.FileHandler('myapp.log')
# 将FileHandler添加到logger中
logger = logging.getLogger()
logger.filter(lambda record: record.levelno == logging.INFO)
logger.addHandler(file_handler)

neg_parser = JsonParser("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\negative\\")
pos_parser = JsonParser("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\positive\\")


def get_method_distribution():
    lst = pos_parser.types
    # 使用Counter统计每个元素出现的次数
    word_counts = Counter(lst)
    # 将出现次数转换为数据框并计算占比
    word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count'])
    word_counts_df['percentage'] = word_counts_df['count'] / word_counts_df['count'].sum()
    # 按照占比降序排序
    word_counts_df = word_counts_df.sort_values('percentage', ascending=False)
    # 输出结果
    # print(word_counts_df)


if __name__ == '__main__':
    # 1. DT  2. SVM  3. NaiveBayes  4. KNN  5. LDA  6. LR  7. K-Means  8. MLP  9. CNN  10.RNN
    # 'charLength', 'astHeight', 'astNodeNumber', 'layoutRelationDataList'
    features = ['occurrences', 'charLength', 'nodeType', "currentLineData"]
    neg_values = neg_parser.get_value(features)
    pos_values = pos_parser.get_value(features)
    sample_num = min(len(neg_values), len(pos_values))
    neg_values = np.array(neg_values)[:sample_num]
    pos_values = np.array(pos_values)[:sample_num]
    X = np.concatenate((neg_values, pos_values))

    get_method_distribution()

    print(pos_parser.cnt)

    logging.info(f"Sample number: {len(X)}")
    y = np.concatenate(
        (np.zeros(len(neg_values)), np.ones(len(pos_values))))

    # 创建支持向量机分类器
    # 定义模型
    # clf = SVC(kernel='linear')
    model_name = "saDecisionTree"
    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier()
        X_scaled_data = X
    elif model_name == 'SVM':
        # 标准化数据 使得每个特征的方差为1，均值为0
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled_data = X_std

        clf = SVC(kernel='linear')
    else:
        # 推荐总数，对的个数，百分比
        indicesT = [idx for idx, val in enumerate(X) if val[0] > 2]
        indicesF = [idx for idx, val in enumerate(X) if val[0] <= 2]
        count = sum(1 for idx in indicesT if y[idx] == 1)
        print("exp出现两次以上, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count, count / len(indicesT)))

        indicesT = [idx for idx, val in enumerate(X) if val[2] == 1]
        indicesF = [idx for idx, val in enumerate(X) if val[2] == 0]
        count = sum(1 for idx in indicesT if y[idx] == 1)
        print("exp包含函数调用, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count, count / len(indicesT)))

        indicesT = [idx for idx, val in enumerate(X) if val[1] > 14]
        indicesF = [idx for idx, val in enumerate(X) if val[1] <= 14]
        count = sum(1 for idx in indicesT if y[idx] == 1)
        print("exp长度超过14个character, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count,
                                                                                   count / len(indicesT)))

        indicesT = [idx for idx, val in enumerate(X) if val[3] >= 41]
        indicesF = [idx for idx, val in enumerate(X) if val[3] < 41]
        count = sum(1 for idx in indicesT if y[idx] == 1)
        print("所在行最长的length不小于41character, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count,
                                                                                              count / len(indicesT)))

        indicesT = [idx for idx, val in enumerate(X) if  val[0] > 1 and val[2] == 1  and val[1] > 25]
        indicesF = [idx for idx, val in enumerate(X) if  val[0] == 1  and val[2] != 1  or val[1] <= 25]
        count = sum(1 for idx in indicesT if y[idx] == 1)
        print(
            "and起来后，对的数据  推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count, count / len(indicesT)))

        # logging.info("Model name error!")
        exit(0)

    # 定义十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []

    # 进行交叉验证
    for fold, (train_index, test_index) in enumerate(kf.split(X_scaled_data)):

        # 划分训练集和测试集
        X_train, X_test = X_scaled_data[train_index], X_scaled_data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # 训练模型
        clf.fit(X_train, y_train)

        # y_predict = clf.predict(test_data)

        # 评估模型
        # score = clf.score(X_test, y_test)
        y_predict = clf.predict(X_test)
        # logging.info(f"SVM Accuracy: {score}")
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(y_predict)):
            if y_test[i] == y_predict[i] and y_predict[i] == 1:
                tp += 1
            elif y_test[i] == y_predict[i] and y_predict[i] == 0:
                tn += 1
            elif y_test[i] != y_predict[i] and y_predict[i] == 1:
                fp += 1
            elif y_test[i] != y_predict[i] and y_predict[i] == 0:
                fn += 1
        accuracy = 0 if tp + tn + fp + fn == 0 else (tp + tn) * 1.0 / (tp + tn + fp + fn)
        precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
        recall = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)
        accuracies.append(round(accuracy * 100, 2))
        precisions.append(round(precision * 100, 2))
        recalls.append(round(recall * 100, 2))
        print(f"Fold {fold + 1}:")
        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print('')

    logging.info(f'model is {model_name}, considering {features}, here are final results: ')
    a = round(np.mean(accuracies) * 1, 2)
    p = round(np.mean(precisions) * 1, 2)
    r = round(np.mean(recalls) * 1, 2)
    f1 = round(2 * p * r / (p + r), 2)
    logging.info(f'accuracy:{a}')
    logging.info(f'precision:{p}')
    logging.info(f'recall:{r}')
    logging.info(f'f1:{f1}')

    tree_rules = export_text(clf, feature_names=features)
    print(tree_rules)

    # names = parser.get_value('occurrences')
