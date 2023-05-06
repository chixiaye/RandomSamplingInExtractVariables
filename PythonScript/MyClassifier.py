import graphviz as graphviz
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text

from JsonParser import JsonParser
import logging

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

if __name__ == '__main__':
    # 1. DT  2. SVM  3. NaiveBayes  4. KNN  5. LDA  6. LR  7. K-Means  8. MLP  9. CNN  10.RNN
    # 'charLength', 'astHeight', 'astNodeNumber', 'layoutRelationDataList'
    features = ['occurrences', 'astHeight', 'astNodeNumber']
    neg_values = neg_parser.get_value(features)
    pos_values = pos_parser.get_value(features)
    sample_num = min(len(neg_values), len(pos_values))
    neg_values = np.array(neg_values)[:sample_num]
    pos_values = np.array(pos_values)[:sample_num]
    X = np.concatenate((neg_values, pos_values))

    logging.info(f"Sample number: {len(X)}")
    y = np.concatenate(
        (np.zeros(len(neg_values)), np.ones(len(pos_values))))

    # 创建支持向量机分类器
    # 定义模型
    # clf = SVC(kernel='linear')
    model_name = "SVM"
    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif model_name == 'SVM':
        clf = SVC(kernel='linear')

    # 定义十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 标准化数据
    # X_scaled_data = X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled_data = X_std

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

    # tree_rules = export_text(clf, feature_names=features)
    # print(tree_rules)

    # names = parser.get_value('occurrences')
