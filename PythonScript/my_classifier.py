# -*- coding: utf-8 -*-
import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, ClusterCentroids, RandomUnderSampler
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

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

neg_parser = JsonParser("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\negative\\", 0)
pos_parser = JsonParser("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\positive\\", 1)

positive_csv_reader = CSVReader('C:\\Users\\30219\\Desktop\\result\\result_4.csv', 1)
negative_csv_reader = CSVReader('C:\\Users\\30219\\Desktop\\result\\result_negative_6.csv', 0)

logging.info('')


# 定义一个函数来递归获取到达叶子节点的条件
def get_node_condition(tree, node_id):
    feature = tree.feature[node_id]
    threshold = tree.threshold[node_id]

    if tree.children_left[node_id] == tree.children_right[node_id]:
        # Reached a leaf node
        class_value = np.argmax(tree.value[node_id])
        condition = f"类别为 {class_value}"
    else:
        condition = f"特征 {feature} <= {threshold}"
        if feature != -2:
            left_child_condition = get_node_condition(tree, tree.children_left[node_id])
            right_child_condition = get_node_condition(tree, tree.children_right[node_id])
            condition = f"({condition} and {left_child_condition}) or ({condition} and {right_child_condition})"

    return condition


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


def get_decision_rules(tree, feature_names, class_names, indent=0):
    rules = []

    def traverse(node_id, operator, threshold, feature_index, samples):
        indent_str = "    " * indent
        if operator == "<=":
            condition = f"{feature_names[feature_index]} <= {threshold}"
        else:
            condition = f"{feature_names[feature_index]} > {threshold}"

        if tree.children_left[node_id] == -1 or tree.children_right[node_id] == -1:
            class_index = np.argmax(tree.value[node_id][0])
            class_name = class_names[class_index]
            rule = f"{indent_str}if ({condition}) {{ return {class_name}; }}"
            rules.append(rule)
        else:
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            traverse(left_child, "<=", tree.threshold[left_child], tree.feature[left_child], samples)
            rule = f"{indent_str}if ({condition}) {{\n"
            rules.append(rule)
            traverse(right_child, ">", tree.threshold[right_child], tree.feature[right_child], samples)
            rule = f"{indent_str}}}"
            rules.append(rule)

    root_node_id = 0
    traverse(root_node_id, "<=", tree.threshold[root_node_id], tree.feature[root_node_id],
             len(tree.value[root_node_id][0]))
    return rules


if __name__ == '__main__':
    # 1. DT  2. SVM  3. NaiveBayes  4. KNN  5. RandomForest  6. LR  7. K-Means  8. MLP  9. CNN  10.RNN
    # 'charLength', 'astHeight', 'astNodeNumber', 'layoutRelationDataList'

    features = ['occurrences', 'charLength', "isGetTypeMethod", 'isArithmeticExpression' ]  # ,'currentLineData'
    # 读取特征数据
    neg_maps = neg_parser.get_value(features)
    pos_maps = pos_parser.get_value(features)

    # 读取ValExtractor的数据， 数据id到可提取的个数的映射关系
    positive_valExtractor_map = positive_csv_reader.read_csv()
    negative_valExtractor_map = negative_csv_reader.read_csv()
    val_extractor_data = {}
    for key in neg_maps.keys():
        val_extractor_data[key] = neg_maps[key][0]
        if key in negative_valExtractor_map:
            value = negative_valExtractor_map[key]
            val_extractor_data[key] = value
            # if value == 'success':
            #     pass
            # elif value == 'error':
            #     pass
            # elif 'fail' in value and value != 'fail':
            #     arr = value[6:-1]
            #     val_extractor_data[key] = value # arr.split('/')[0]
            #     print(arr.split('/'))
    for key in pos_maps.keys():
        val_extractor_data[key] = pos_maps[key][0]
        if key in positive_valExtractor_map:
            value = positive_valExtractor_map[key]
            val_extractor_data[key] = value
            # if value == 'success':
            #     pass
            # elif value == 'error':
            #     pass
            # elif 'fail' in value and value != 'fail':
            #     arr = value[6:-1]
            #     val_extractor_data[key] = value  # arr.split('/')[0]
            #     print(arr.split('/'))

    # map总的数据到每条数据id的映射关系
    index_to_data_map = {}
    neg_value_list = []
    pos_value_list = []
    for key in neg_maps.keys():
        index_to_data_map[len(index_to_data_map)] = key
        neg_value_list.append(neg_maps[key])
    for key in pos_maps.keys():
        index_to_data_map[len(index_to_data_map)] = key
        pos_value_list.append(pos_maps[key])
    # sample_num = min(len(neg_value_list), len(pos_value_list))
    neg_values = np.array(neg_value_list)[:len(neg_value_list)]
    pos_values = np.array(pos_value_list)[:len(pos_value_list)]
    X = np.concatenate((neg_values, pos_values))

    # get_method_distribution()

    # print(pos_parser.cnt)

    logging.info(f"Sample number: {len(X)}")
    y = np.concatenate(
        (np.zeros(len(neg_values)), np.ones(len(pos_values))))

    # 创建支持向量机分类器
    # 定义模型
    # clf = SVC(kernel='linear')
    model_name = "DecisionTree"
    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier( max_depth=4 )  # ,class_weight={0:1,1:37}
        # clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, min_samples_split=10)  #
        # clf = DecisionTreeClassifier(   )  #
    elif model_name == 'SVM':
        # 标准化数据 使得每个特征的方差为1，均值为0
        # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # X_scaled_data = X_std
        clf = SVC()  # kernel='linear'
    elif model_name == 'NaiveBayes':
        clf = BernoulliNB()  # 标准化转换
    elif model_name == 'KNN':
        clf = KNeighborsClassifier()  # 标准化转换
    elif model_name == 'K-Means':
        clf = KMeans(2, random_state=0)  # 标准化转换
    elif model_name == 'MLP':
        clf = MLPClassifier()  # 标准化转换
    elif model_name == 'LR':
        clf = LogisticRegression()  # 标准化转换
    elif model_name == 'RandomForest':
        X_scaled_data = X
        # scaler = StandardScaler()  # 标准化转换
        # scaler.fit(X)  # 训练标准化对象
        # X_scaled_data = scaler.transform(X_scaled_data)  # 转换数据集
        clf = RandomForestClassifier()  # 标准化转换 max_depth=3, n_estimators=1000

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

        indicesT = [idx for idx, val in enumerate(X) if val[4] >= 21]
        indicesF = [idx for idx, val in enumerate(X) if val[4] < 21]
        count = sum(1 for idx in indicesT if y[idx] == 1)
        print("所在行最长的length不小于21character, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count,
                                                                                              count / len(indicesT)))

        indicesT = [idx for idx, val in enumerate(X) if val[0] > 2 and val[2] == 1 and val[1] > 25]
        indicesF = [idx for idx, val in enumerate(X) if val[0] == 1 and val[2] != 1 or val[1] <= 25]
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
    tp_and_fp = []
    tps = []

    X_scaled_data = X
    cnt_lg = 0
    cnt_sm = 0
    # 进行交叉验证
    for fold, (train_index, test_index) in enumerate(kf.split(X_scaled_data)):

        # 划分训练集和测试集 copy 不改写原有数据
        X_train, X_test = X_scaled_data[train_index].copy(), X_scaled_data[test_index].copy()
        y_train, y_test = y[train_index].copy(), y[test_index].copy()

        # # 实例化SMOTE对象
        # smote = SMOTE(random_state=1)
        # # 进行过采样
        # X_train , y_train = smote.fit_resample(X_train , y_train)

        # 通过对多数类样本进行有放回或无放回地随机采样来选择部分多数类样本。
        cc = RandomUnderSampler(random_state=42)
        before = len(X_train)
        X_train, y_train = cc.fit_resample(X_train, y_train)
        after = len(X_train)
        print('Removed {} samples'.format(before - after))

        # 对训练集进行标准化
        scaler = StandardScaler()  # 标准化转换
        scaler.fit(X_train)  # 标准化训练集 对象
        X_train_norm = scaler.transform(X_train)  # 转换训练集
        X_test_norm = scaler.transform(X_test)  # 转换测试集

        # 训练模型
        clf.fit(X_train_norm, y_train)

        # y_predict = clf.predict(test_data)

        X_test_copy = X_test.copy()
        # 评估模型
        # score = clf.score(X_test, y_test)

        for index in range(0, len(X_test)):
            if 'occurrences' in features:
                # 判断对象是否为数组
                if isinstance(val_extractor_data[index_to_data_map[test_index[index]]], list):
                    # print(val_extractor_data[index_to_data_map[test_index[index]]])
                    X_test_copy[index][0] = val_extractor_data[index_to_data_map[test_index[index]]][1]
                else:
                    X_test_copy[index][0] = val_extractor_data[index_to_data_map[test_index[index]]]

        X_test_norm_copy = scaler.transform(X_test_copy)  # 转换测试集
        y_predict = clf.predict(X_test_norm_copy)
        for index in range(0, len(X_test)):
            # 如果预测为正 送入ValExtractor检验
            if y_predict[index] == 1:
                tmp = X_test_copy[index].copy()  # 判断对象是否为数组
                if isinstance(val_extractor_data[index_to_data_map[test_index[index]]], list):
                    tmp[0] = val_extractor_data[index_to_data_map[test_index[index]]][2]
                tmp_norm = scaler.transform([tmp])  # 重新转换测试集
                y_predict[index] = clf.predict(tmp_norm)[0]
                if y_predict[index] == 0:
                    print(
                        index_to_data_map[test_index[index]] + "," + "valextractor" + ":" + str(
                            tmp[0]) + "," + "original:" +
                        str(X_test_copy[index][0]))
                pass

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
        # accuracy_positive  就是总样本的recall
        # precision_positive  在正样本中 不存在把负样本错误地预测为正 所以为1
        # recall_positive  在正样本中, 就是正样本中有多少被预测到了 就是总样本的recall
        # accuracy_negative   FN/(TN + FP)
        # precision_negative  在负样本中 不存在把正样本成功预测为正 所以为0
        # recall_negative  在负样本中 不存在把正样本成功预测为正 所以为0
        recall_positive = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)
        accuracies.append(round(accuracy * 100, 2))
        precisions.append(round(precision * 100, 2))
        recalls.append(round(recall * 100, 2))
        tps.append(tp)
        tp_and_fp.append(tp + fp)
        print(f"Fold {fold + 1}:")
        print(f'precision: {round(precision * 100, 2)}')
        print(f'recall: {round(recall * 100, 2)}')
        print(f'accuracy: {round(accuracy * 100, 2)}')
        print(f'f1: {round(2 * precision * recall / (precision + recall) * 100, 2)}')
        # print(f"pos acc {round(tp/(tp+ fn)* 100, 2)}:")
        # print(f'{tp + fp} {tp}')
        print('')

    logging.info(f' model is {model_name}, considering {features}, here are final results: ')  # {clf.get_depth()}
    a = round(np.mean(accuracies) * 1, 2)
    p = round(np.mean(precisions) * 1, 2)
    r = round(np.mean(recalls) * 1, 2)
    f1 = round(2 * p * r / (p + r), 2)
    logging.info(f'precision:{p}')
    logging.info(f'recall:{r}')
    logging.info(f'accuracy:{a}')
    logging.info(f'f1:{f1}')
    logging.info(f'推荐总数{sum(tp_and_fp)}, 对的个数{sum(tps)} ')


    if model_name == 'DecisionTree':
        clf.fit(X_scaled_data, y)
        y_pred = clf.predict(X_scaled_data)
        y_test = y
        tree_rules = export_text(clf, feature_names=features)
        # 指定图幅大小
        plt.figure(figsize=(30, 35), dpi=200)
        _ = tree.plot_tree(clf, fontsize=10, feature_names=features, filled=True, rounded=True, class_names=['0', '1'])
        # plt.figure(figsize=(15, 10), dpi=600)
        print("plotting decision tree...")
        plt.show()
        plt.savefig('./resource/decision_tree.png', format='png')

        # 获取叶子节点的索引
        leaf_indices = clf.apply(X_scaled_data)

        # 初始化一个空字典来保存每个叶子节点的 precision
        leaf_precisions = {}

        # # 指定叶子节点的索引
        # leaf_index = 4
        #
        # # 获取到达叶子节点的条件
        # condition = get_node_condition(clf.tree_, leaf_index)
        # print(f"到达叶子节点 {leaf_index} 的条件为: {condition}")

        # 遍历每个叶子节点
        for leaf_index in np.unique(leaf_indices):
            # 获取当前叶子节点的预测结果和真实标签
            leaf_y_pred = y_pred[leaf_indices == leaf_index]
            leaf_y_true = y_test[leaf_indices == leaf_index]

            tp, fp, tn, fn = 0, 0, 0, 0
            for i in range(len(leaf_y_pred)):
                if leaf_y_true[i] == leaf_y_pred[i] and leaf_y_pred[i] == 1:
                    tp += 1
                elif leaf_y_true[i] == leaf_y_pred[i] and leaf_y_pred[i] == 0:
                    tn += 1
                elif leaf_y_true[i] != leaf_y_pred[i] and leaf_y_pred[i] == 1:
                    fp += 1
                elif leaf_y_true[i] != leaf_y_pred[i] and leaf_y_pred[i] == 0:
                    fn += 1

            if tp + fp == 0:
                continue

            precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
            pstr = f"推荐{tp + fp}, 对了{tp}, 正确率{precision}"  # precision_score(leaf_y_true, leaf_y_pred)

            # 将 precision 存储到字典中
            leaf_precisions[leaf_index] = pstr

        # 打印每个叶子节点的 precision
        for leaf_index, precision in leaf_precisions.items():
            print(f"Leaf {leaf_index}: Precision = {precision}")
        # print(X_test)
        # print(clf.predict([[1, 15, 0, 0]]))
