import logging
import os

import keras
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.src.optimizers.adam import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow import random
from tensorflow.python.keras.saving.save import load_model
from imblearn.over_sampling import SMOTE, ADASYN
from CSVReader import CSVReader
from JsonParser import JsonParser


RANDOM_STATE = 42

# 设置日志级别为INFO，只记录INFO级别以上的信息
logging.basicConfig(level=logging.INFO)
# 创建FileHandler并配置日志文件名
file_handler = logging.FileHandler('myapp.log')
# 将FileHandler添加到logger中
logger = logging.getLogger()
logger.filter(lambda record: record.levelno == logging.INFO)
logger.addHandler(file_handler)

neg_parser = JsonParser(["C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\final-data\\negative"],
                        0,
                        0)  # ,"C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\negative\\"
pos_parser = JsonParser(["C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\final-data\\positive"],
                        0,
                        1)  # ,"C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\positive\\"

positive_csv_reader = CSVReader('C:\\Users\\30219\\Desktop\\result\\result-v1.csv', 1)
negative_csv_reader = CSVReader('C:\\Users\\30219\\Desktop\\result\\result-neg-v1.csv', 0)

logging.info('')

features = ['occurrences', 'maxEndColumnNumberInCurrentLine', 'charLength', 'tokenLength',
            'numsParentVariableDeclarationFragment', 'isName', 'isLiteral', 'isGetMethod',
            'isArithmeticExp', 'largestLineGap',
            'numsParentArithmeticExp', 'isMethodInvocation']

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
for key in pos_maps.keys():
    val_extractor_data[key] = pos_maps[key][0]
    if key in positive_valExtractor_map:
        value = positive_valExtractor_map[key]
        val_extractor_data[key] = value

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

logging.info(f"Sample number: {len(X)}")
y = np.concatenate(
    (np.zeros(len(neg_values)), np.ones(len(pos_values))))

# 定义十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
tp_and_fp = []
tps = []
X_scaled_data = X

np.random.seed(42)  # 固定随机种子，使每次运行结果固定
random.set_seed(42)

# 进行交叉验证
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled_data)):

    # 划分训练集和测试集 copy 不改写原有数据
    X_train, X_test = X_scaled_data[train_index].copy(), X_scaled_data[test_index].copy()
    y_train, y_test = y[train_index].copy(), y[test_index].copy()

    # ADASYN（Adaptive Synthetic Sampling） 对顺序不敏感
    sampler = ADASYN(sampling_strategy='auto', random_state=RANDOM_STATE)
    # 进行过采样
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    # 对训练集进行标准化
    scaler = StandardScaler()  # 标准化转换
    scaler.fit(X_train)  # 标准化训练集 对象
    X_train_norm = scaler.transform(X_train)  # 转换训练集
    X_test_norm = scaler.transform(X_test)  # 转换测试集


    model_path = './cnn/model_' + str(fold) + '.h5'
    # 创建模型结构：输入层的特征维数为len features；1层k个神经元的relu隐藏层；线性的输出层；
    k = 50  # 最佳神经元数
    if os.path.isfile(model_path):
        model = keras.models.load_model(model_path, custom_objects={'BatchNormalization': BatchNormalization, 'Adam': Adam},compile = False)
    else:
        model = Sequential()

        model.add(BatchNormalization(input_dim=len(features)))  # 输入层 批标准化

        model.add(Dense(k,
                        kernel_initializer='random_uniform',  # 均匀初始化
                        activation='relu',  # relu激活函数
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),  # L1及L2 正则项
                        use_bias=True))  # 隐藏层

        model.add(Dropout(0.05))  # dropout法  防止过拟合
        model.add(Dense(1, activation="sigmoid", use_bias=True))  # 输出层 介于0和1之间的概率值
        model.compile(optimizer='adam', loss='mse')
        # 训练模型
        history = model.fit(X_train_norm,
                            y_train,
                            epochs=500,  # 训练迭代次数
                            batch_size=64,  # 每epoch采样的batch大小
                            validation_split=0.1,  # 从训练集再拆分验证集，作为早停的衡量指标
                            callbacks=[EarlyStopping(monitor='val_loss', patience=20)],  # 早停法
                            verbose=False)  # 不输出过程
        model.save(model_path)

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

    res = model.predict(X_test_norm_copy, verbose=False)
    # 评估模型
    y_predict = [1 if pred > 0.5 else 0 for pred in res]
    for index in range(0, len(X_test)):
        # 如果预测为正 送入ValExtractor检验
        if y_predict[index] == 1:
            # tmp = X_test_copy[index].copy()  # 判断对象是否为数组
            # if isinstance(val_extractor_data[index_to_data_map[test_index[index]]], list):
            #     tmp[0] = val_extractor_data[index_to_data_map[test_index[index]]][2]
            # tmp_norm = scaler.transform([tmp])  # 重新转换测试集
            # y_predict[index] = 1 if model.predict(tmp_norm, verbose=False)[0] > 0.5 else 0
            # if y_predict[index] == 0:
            #     print(index_to_data_map[test_index[index]], tmp[0], X_test_copy[index][0])
            pass

    # test_loss, test_acc = model.evaluate(X_test_norm, y_test)
    # print('Test accuracy:', test_acc)

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
    print(f'f1: { 0 if (precision + recall)==0 else round(2 * precision * recall / (precision + recall) * 100, 2)}')
    # print(f'{tp + fp} {tp}')
    print('')

logging.info(f' model is cnn, considering {features}, here are final results: ')  # {clf.get_depth()}
a = round(np.mean(accuracies) * 1, 2)
p = round(np.mean(precisions) * 1, 2)
r = round(np.mean(recalls) * 1, 2)
f1 = round(2 * p * r / (p + r), 2)
logging.info(f'precision:{p}')
logging.info(f'recall:{r}')
logging.info(f'accuracy:{a}')
logging.info(f'f1:{f1}')
logging.info(f'推荐总数{sum(tp_and_fp)}, 对的个数{sum(tps)} ')
