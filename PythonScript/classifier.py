import os

import joblib
import numpy as np
import pymysql
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, SimpleRNN
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from JsonParser import JsonParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
db = pymysql.connect(user='root', password='123456', host='localhost', database='icse', port=3306, charset='utf8')
cursor = db.cursor()


def get_train():
    # 选择决策预测表中的所有数据，其中重构 ID 在重构标签表中也存在
    train_sql = "select * from decision_predict where refactor_id in (select refactor_id from refactoring_label)"
    # 选择重构标签表中的所有数据
    label_sql = "select * from refactoring_label"
    # 执行 SQL 查询语句
    cursor.execute(train_sql)
    # 获取查询结果
    results = cursor.fetchall()
    # 执行 SQL 查询语句
    cursor.execute(label_sql)
    # 获取查询结果
    labels = cursor.fetchall()
    # 创建空字典
    dictionary = {}
    # 遍历标签结果
    for label in labels:
        # 将标签名称作为键，标签 ID 作为值添加到字典中
        dictionary[label[1]] = label[2]
    # 创建空列表，用于存储训练数据和标签
    train_data = []
    train_label = []
    # 遍历决策预测结果
    for row in results:
        # 调用 get_feature 函数获取行特征，并将其添加到训练数据列表中
        train_data.append(get_feature(row))
        # 通过行的重构类型名称从字典中获取标签 ID，并将其添加到训练标签列表中
        train_label.append(dictionary[row[9]])
    # 返回训练数据和标签
    return train_data, train_label


def get_test():
    # 从 decision_predict 表中选择 refactor_id 在 refactoring_label 表中存在的所有行
    test_sql = "select * from decision_predict where refactor_id in (select refactor_id from refactoring_label)"
    label_sql = "select * from refactoring_label"
    cursor.execute(test_sql)
    results = cursor.fetchall()  # 获取所有结果
    cursor.execute(label_sql)
    labels = cursor.fetchall()  # 获取所有标签
    dictionary = {}
    # 构建一个字典，键为 label 表的第二列，值为 label 表的第三列
    for label in labels:
        dictionary[label[1]] = label[2]
    test_data = []
    test_label = []
    # 对结果进行遍历，将每一行的特征值和标签值加入到 test_data 和 test_label 列表中
    for row in results:
        test_data.append(get_feature(row))
        test_label.append(dictionary[row[9]])

    # 从 decision_filter_predict 表中选择 refactor_id 在 filter_label 表中存在的所有行
    test_sql = "select * from decision_filter_predict where refactor_id in (select refactor_id from filter_label)"
    label_sql = "select * from filter_label"
    cursor.execute(test_sql)
    results = cursor.fetchall()
    cursor.execute(label_sql)
    labels = cursor.fetchall()
    dictionary = {}
    # 构建一个字典，键为 label 表的第二列，值为 label 表的第三列
    for label in labels:
        dictionary[label[1]] = label[2]
    # 对结果进行遍历，将每一行的特征值和标签值加入到 test_data 和 test_label 列表中
    for row in results:
        test_data.append(get_feature(row))
        test_label.append(dictionary[row[9]])
    return test_data, test_label


def get_feature(row):
    # 定义一个空列表，用于存储特征值
    feature = []
    # 获取原始代码调用次数、移动代码调用次数、匹配代码调用次数、原始代码元素数、移动代码元素数、匹配代码元素数
    original_invoke_nums = row[1]
    moved_invoke_nums = row[2]
    matched_invoke_nums = row[3]
    original_code_elements = row[4]
    moved_code_elements = row[5]
    matched_code_elements = row[6]
    # 计算并添加特征值：匹配代码调用次数占原始代码调用次数的比例
    feature.append(0 if matched_invoke_nums == 0 else float(matched_invoke_nums) / float(original_invoke_nums))
    # 计算并添加特征值：匹配代码调用次数占移动代码调用次数的比例
    feature.append(0 if matched_invoke_nums == 0 else float(matched_invoke_nums) / float(moved_invoke_nums))
    # 添加特征值：匹配代码元素数
    feature.append(int(matched_code_elements))
    # 计算并添加特征值：匹配代码元素数占原始代码元素数的比例
    feature.append(0 if original_code_elements == 0 else float(matched_code_elements) / float(original_code_elements))
    # 计算并添加特征值：匹配代码元素数占移动代码元素数的比例
    feature.append(0 if moved_code_elements == 0 else float(matched_code_elements) / float(moved_code_elements))
    # 返回特征值列表
    return feature


def main(opt):
    train_data, train_label = get_train()
    test_data, y_hat = get_test()
    accuracies = []
    precisions = []
    recalls = []
    for i in range(10):
        if opt == 1:
            dt = DecisionTreeClassifier(criterion='gini', max_features=len(train_data[0]))
            dt.fit(train_data, train_label)
            y_predict = dt.predict(test_data)
        elif opt == 2:
            svm = SVC()
            svm.fit(train_data, train_label)
            y_predict = svm.predict(test_data)
        elif opt == 3:
            bnb = BernoulliNB()
            bnb.fit(train_data, train_label)
            y_predict = bnb.predict(test_data)
        elif opt == 4:
            knn = KNeighborsClassifier(n_neighbors=2)
            knn.fit(train_data, train_label)
            y_predict = knn.predict(test_data)
        elif opt == 5:
            lda = LinearDiscriminantAnalysis()
            lda.fit(train_data, train_label)
            y_predict = lda.predict(test_data)
        elif opt == 6:
            lr = LogisticRegression()
            lr.fit(train_data, train_label)
            y_predict = lr.predict(test_data)
        elif opt == 7:
            km = KMeans(n_clusters=2)
            km.fit(train_data, train_label)
            y_predict = km.predict(test_data)
        elif opt == 8:
            train_x = np.array(train_data)
            train_y = to_categorical(np.array(train_label), num_classes=2)
            test_x = np.array(test_data)
            model = Sequential()
            model.add(Dense(128, input_dim=train_x.shape[1], activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(2, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(train_x, train_y, epochs=5, verbose=0)
            detector = model.predict(test_x)
            y_predict = [np.argmax(detector[i]) for i in range(len(detector))]
        elif opt == 9:
            train_data = np.asfarray(train_data)
            train_x = np.expand_dims(train_data, axis=2)
            train_y = to_categorical(np.array(train_label), num_classes=2)
            test_data = np.asfarray(test_data)
            test_x = np.expand_dims(test_data, axis=2)
            model = Sequential()
            model.add(Conv1D(128, 1, input_shape=(train_x.shape[1], 1), padding='same', activation='tanh'))
            model.add(Conv1D(128, 1, activation='tanh'))
            model.add(Conv1D(128, 1, activation='tanh'))
            model.add(Flatten())
            model.add(Dense(2, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(train_x, train_y, epochs=5, verbose=0)
            detector = model.predict(test_x)
            y_predict = [np.argmax(detector[i]) for i in range(len(detector))]
        elif opt == 10:
            train_data = np.asfarray(train_data)
            train_x = np.expand_dims(train_data, axis=2)
            train_y = to_categorical(np.array(train_label), num_classes=2)
            test_data = np.asfarray(test_data)
            test_x = np.expand_dims(test_data, axis=2)
            model = Sequential()
            model.add(SimpleRNN(128, input_shape=(train_x.shape[1], 1), activation='tanh', return_sequences=True))
            model.add(Dropout(0.2))
            model.add(SimpleRNN(128, activation='tanh', return_sequences=True))
            model.add(Dropout(0.2))
            model.add(SimpleRNN(128, activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(2, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(train_x, train_y, epochs=5, verbose=0)
            detector = model.predict(test_x)
            y_predict = [np.argmax(detector[i]) for i in range(len(detector))]

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(y_predict)):
            if y_hat[i] == y_predict[i] and y_predict[i] == 1:
                tp += 1
            elif y_hat[i] == y_predict[i] and y_predict[i] == 0:
                tn += 1
            elif y_hat[i] != y_predict[i] and y_predict[i] == 1:
                fp += 1
            elif y_hat[i] != y_predict[i] and y_predict[i] == 0:
                fn += 1
        accuracy = 0 if tp + tn + fp + fn == 0 else (tp + tn) * 1.0 / (tp + tn + fp + fn)
        precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
        recall = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)
        accuracies.append(round(accuracy * 100, 2))
        precisions.append(round(precision * 100, 2))
        recalls.append(round(recall * 100, 2))
    a = round(np.mean(accuracies) * 1, 2)
    p = round(np.mean(precisions) * 1, 2)
    r = round(np.mean(recalls) * 1, 2)
    f1 = round(2 * p * r / (p + r), 2)
    print('accuracy:', a)
    print('precision:', p)
    print('recall:', r)
    print('f1:', f1)


# def mark():
#     mark_sql = "select * from decision_predict where refactor_id not in (select refactor_id from refactoring_label)"
#     data = []
#     ids = []
#     cursor.execute(mark_sql)
#     results = cursor.fetchall()
#     for row in results:
#         feature = []
#         original_invoke_nums = row[1]
#         moved_invoke_nums = row[2]
#         matched_invoke_nums = row[3]
#         original_code_elements = row[4]
#         moved_code_elements = row[5]
#         matched_code_elements = row[6]
#         refactor_id = row[9]
#         feature.append(0 if matched_invoke_nums == 0 else float(matched_invoke_nums) / float(original_invoke_nums))
#         feature.append(0 if matched_invoke_nums == 0 else float(matched_invoke_nums) / float(moved_invoke_nums))
#         feature.append(int(matched_code_elements))
#         feature.append(float(matched_code_elements) / float(original_code_elements))
#         feature.append(float(matched_code_elements) / float(moved_code_elements))
#         data.append(feature)
#         ids.append(refactor_id)
#     train_data, train_label = get_train()
#     test_data, y_hat = get_test()
#     dt = DecisionTreeClassifier(criterion='gini', max_features=len(train_data[0]))
#     dt.fit(train_data, train_label)
#     y_predict = dt.predict(test_data)
#     tp, fp, tn, fn = 0, 0, 0, 0
#     for i in range(len(y_predict)):
#         if y_hat[i] == y_predict[i] and y_predict[i] == 1:
#             tp += 1
#         elif y_hat[i] == y_predict[i] and y_predict[i] == 0:
#             tn += 1
#         elif y_hat[i] != y_predict[i] and y_predict[i] == 1:
#             fp += 1
#         elif y_hat[i] != y_predict[i] and y_predict[i] == 0:
#             fn += 1
#     precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
#     recall = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)
#     p = round(precision * 100, 2)
#     r = round(recall * 100, 2)
#     f1 = round(2 * p * r / (p + r), 2)
#     print('precision:', p)
#     print('recall:', r)
#     print('f1:', f1)
#     y_predict = list(dt.predict(data))
#     label_sql = "select * from refactoring_label"
#     cursor.execute(label_sql)
#     results = cursor.fetchall()
#     for row in results:
#         y_predict.append(row[2])
#         ids.append(row[1])
#     update_sql = "update decision_predict set predict_label = %s where refactor_id = %s"
#     for i in range(len(y_predict)):
#         content = (y_predict[i], ids[i])
#         cursor.execute(update_sql, content)
#         db.commit()

def mark():
    mark_sql = "select * from decision_filter_predict"
    data = []
    ids = []
    cursor.execute(mark_sql)
    results = cursor.fetchall()
    for row in results:
        feature = []
        original_invoke_nums = row[1]
        moved_invoke_nums = row[2]
        matched_invoke_nums = row[3]
        original_code_elements = row[4]
        moved_code_elements = row[5]
        matched_code_elements = row[6]
        refactor_id = row[9]
        feature.append(0 if matched_invoke_nums == 0 else float(matched_invoke_nums) / float(original_invoke_nums))
        feature.append(0 if matched_invoke_nums == 0 else float(matched_invoke_nums) / float(moved_invoke_nums))
        feature.append(int(matched_code_elements))
        feature.append(
            0 if original_code_elements == 0 else float(matched_code_elements) / float(original_code_elements))
        feature.append(0 if moved_code_elements == 0 else float(matched_code_elements) / float(moved_code_elements))
        data.append(feature)
        ids.append(refactor_id)
    dt = joblib.load('dt.pkl')
    y_predict = list(dt.predict(data))
    update_sql = "update decision_filter_predict set predict_label = %s where refactor_id = %s"
    for i in range(len(y_predict)):
        content = (y_predict[i], ids[i])
        cursor.execute(update_sql, content)
        db.commit()


if __name__ == '__main__':
    # 1. DT  2. SVM  3. NaiveBayes  4. KNN  5. LDA  6. LR  7. K-Means  8. MLP  9. CNN  10.RNN
    # mark()
    # main(1)
    # print(123)
    parser = JsonParser("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\negative\\")
    names = parser.get_value('id')
    print(names)
