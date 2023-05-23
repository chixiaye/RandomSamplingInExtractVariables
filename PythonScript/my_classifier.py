import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


if __name__ == '__main__':
    # 1. DT  2. SVM  3. NaiveBayes  4. KNN  5. LDA  6. LR  7. K-Means  8. MLP  9. CNN  10.RNN
    # 'charLength', 'astHeight', 'astNodeNumber', 'layoutRelationDataList'
    features = ['occurrences', 'charLength', 'isArithmeticExpression', "isGetTypeMethod"]  # ,'currentLineData'
    neg_values = neg_parser.get_value(features)
    pos_values = pos_parser.get_value(features)
    sample_num = min(len(neg_values), len(pos_values))
    neg_values = np.array(neg_values)[:sample_num]
    pos_values = np.array(pos_values)[:sample_num]

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
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, min_samples_split=2)  #
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
        tps.append(tp)
        tp_and_fp.append(tp + fp)
        print(f"Fold {fold + 1}:")
        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print('')

    logging.info(f' {clf.get_depth()} model is {model_name}, considering {features}, here are final results: ')
    a = round(np.mean(accuracies) * 1, 2)
    p = round(np.mean(precisions) * 1, 2)
    r = round(np.mean(recalls) * 1, 2)
    f1 = round(2 * p * r / (p + r), 2)
    logging.info(f'accuracy:{a}')
    logging.info(f'precision:{p}')
    logging.info(f'recall:{r}')
    logging.info(f'f1:{f1}')
    logging.info(f'推荐总书{sum(tp_and_fp)}, 对的个数{sum(tps)}')

    # tree_rules = export_text(clf, feature_names=features)

    clf.fit(X_scaled_data, y)
    y_pred = clf.predict(X_scaled_data)
    y_test = y

    plt.figure(figsize=(50, 80))
    tree.plot_tree(clf, feature_names=features, filled=True, rounded=True, class_names=['0', '1'])
    # plt.show()
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

    # names = parser.get_value('occurrences')
