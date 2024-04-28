# -*- coding: utf-8 -*-
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

# import pybaobabdt
# import pygraphviz as pgv

from CSVReader import CSVReader
from JsonParser import JsonParser

RANDOM_STATE = 43

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


def cohen_d(group1, group2):
    mean_diff = sum(group1) / len(group1) - sum(group2) / len(group2)
    pooled_stdev = ((sum((x - mean_diff) ** 2 for x in group1) + sum((x - mean_diff) ** 2 for x in group2)) / (
            len(group1) + len(group2) - 2)) ** 0.5 if len(group1) + len(group2) - 2 != 0 else 0
    if pooled_stdev == 0:
        return 0

    return mean_diff / pooled_stdev


if __name__ == '__main__':
    # 1. DT  2. SVM  3. NaiveBayes  4. KNN  5. RandomForest  6. LR  7. K-Means  8. MLP  9. CNN  10.RNN
    # 'charLength', 'astHeight', 'astNodeNumber', 'layoutRelationDataList', 'isSimpleName'  'isClassInstanceCreation',
    # 'isLambdaExpression','sumLineGap','numInitializerInlocationInParent'
    # 'maxStartColumnNumberIncurrentLineData',,'isArrayAccess'
    # 假设排除 'charLength',"isGetMethod",'numsParentReturnStatement','numsInCond', 'isQualifiedName', 'isArithmeticExp','tokenLength', 'isStreamMethod',
    # case study 派出的特征  'numsParentArithmeticExp','tokenLength','isStreamMethod','numsParentCall', 'astHeight',
    # 全部特征
    # features = ['occurrences', 'maxEndColumnNumberInCurrentLine', 'currentLineData', 'charLength', 'tokenLength',
    #                 'numsParentVariableDeclarationFragment',
    #                 'isSimpleName', 'isQualifiedName', 'isNumberLiteral', 'isCharacterLiteral', 'isStringLiteral',
    #                 "isGetMethod", 'isArithmeticExp', 'isClassInstanceCreation', 'isMethodInvocation', 'isStreamMethod',
    #                 'numsParentThrowStatement', 'numsParentReturnStatement', 'numsInCond', 'numsInAssignment',
    #                 'largestLineGap',
    #                 'maxParentAstHeight', 'numsParentArithmeticExp', 'numsParentCall',]  # 'currentLineData',
    features = ['occurrences', 'maxEndColumnNumberInCurrentLine', 'charLength', 'tokenLength',
                'numsParentVariableDeclarationFragment', 'isName', 'isLiteral', 'isGetMethod',
                'isArithmeticExp',  'largestLineGap',
                'numsParentArithmeticExp','isMethodInvocation'] # , 'numsParentCall'   'maxParentAstHeight',   'isClassInstanceCreation',  'numsInCond',
    feature_elimination_experiment_enable = True

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

    # 创建支持向量机分类器
    # 定义模型
    # clf = SVC(kernel='linear')
    model_name = "DecisionTree" #"DecisionTree"
    if model_name == 'DecisionTree':
        # path = model_gini.cost_complexity_pruning_path(X_train, y_train)
        # ccp_alphas, impurities = path.ccp_alphas, path.impurities

        clf = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=10,
                                     random_state=RANDOM_STATE)  # class_weight={0:1,1:32} min_samples_leaf=5, min_samples_split=10,
        # clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, min_samples_split=10)  #
        # clf = DecisionTreeClassifier(   )  #
    elif model_name == 'SVM':
        # 标准化数据 使得每个特征的方差为1，均值为0
        # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # X_scaled_data = X_std
        clf =  SVC(random_state=RANDOM_STATE)  # 标准化转换
    elif model_name == 'NaiveBayes':
        clf = BernoulliNB( )  # 标准化转换
    elif model_name == 'KNN':
        clf = KNeighborsClassifier( )  # 标准化转换
    elif model_name == 'K-Means':
        clf = KMeans(2, random_state=RANDOM_STATE)  # 标准化转换
    elif model_name == 'MLP':
        clf = MLPClassifier(random_state=RANDOM_STATE)  # 标准化转换
    elif model_name == 'LR':
        clf = LogisticRegression()  # 标准化转换
    elif model_name == 'RandomForest':
        # X_scaled_data = X
        # scaler = StandardScaler()  # 标准化转换
        # scaler.fit(X)  # 训练标准化对象
        # X_scaled_data = scaler.transform(X_scaled_data)  # 转换数据集
        clf = RandomForestClassifier(random_state=RANDOM_STATE)  # 标准化转换 max_depth=3, n_estimators=1000

    else:
        # 推荐总数，对的个数，百分比
        # indicesT = [idx for idx, val in enumerate(X) if val[0] > 2]
        # indicesF = [idx for idx, val in enumerate(X) if val[0] <= 2]
        # count = sum(1 for idx in indicesT if y[idx] == 1)
        # print("exp出现两次以上, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count, count / len(indicesT)))
        #
        # indicesT = [idx for idx, val in enumerate(X) if val[2] == 1]
        # indicesF = [idx for idx, val in enumerate(X) if val[2] == 0]
        # count = sum(1 for idx in indicesT if y[idx] == 1)
        # print("exp包含函数调用, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count, count / len(indicesT)))
        #
        # indicesT = [idx for idx, val in enumerate(X) if val[1] > 14]
        # indicesF = [idx for idx, val in enumerate(X) if val[1] <= 14]
        # count = sum(1 for idx in indicesT if y[idx] == 1)
        # print("exp长度超过14个character, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count,
        #                                                                            count / len(indicesT)))

        # indicesT = [idx for idx, val in enumerate(X) if val[4] >= 21]
        # indicesF = [idx for idx, val in enumerate(X) if val[4] < 21]
        # count = sum(1 for idx in indicesT if y[idx] == 1)
        # print("所在行最长的length不小于21character, 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count,
        #                                                                                       count / len(indicesT)))

        # indicesT = [idx for idx, val in enumerate(X) if
        #             val[0] != 1 and (val[2] == 1 or val[3] == 1) and val[4] == 0 and val[5] == 0 and val[6] == 0 and
        #             val[7] == 0 and val[8] == 0]
        # indicesF = [idx for idx, val in enumerate(X) if not (
        #         val[0] != 1 and (val[2] == 1 or val[3] == 1) and val[4] == 0 and val[5] == 0 and val[6] == 0 and
        #         val[7] == 0 and val[8] == 0)]
        # count = sum(1 for idx in indicesT if y[idx] == 1)
        # print(" 推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count, count / len(indicesT)))

        # indicesT = [idx for idx, val in enumerate(X) if val[0] > 2 and val[2] == 1 and val[1] > 25]
        # indicesF = [idx for idx, val in enumerate(X) if val[0] == 1 and val[2] != 1 or val[1] <= 25]
        # count = sum(1 for idx in indicesT if y[idx] == 1)
        # print(
        #     "and起来后，对的数据  推荐总数 {}，对的个数 {}，百分比 {}".format(len(indicesT), count, count / len(indicesT)))

        # logging.info("Model name error!")
        if 1 == 1:
            # 按照项目名聚合feature
            case_study_data_pos = []
            case_study_data_neg = []
            for project_name in pos_parser.project_index_map.keys():
                neg_feature_values = []
                for v in neg_parser.project_index_map[project_name]:
                    neg_feature_values.append(neg_maps[v.replace('.json', '') + '_' + str(0)])
                pos_feature_values = []
                for v in pos_parser.project_index_map[project_name]:
                    pos_feature_values.append(pos_maps[v.replace('.json', '') + '_' + str(1)])
                significant_features_count = 0
                # 以 X1 为基准，逐个特征进行 t 检验
                for i in range(len(features)):
                    X1_feature = [sample[i] for sample in neg_feature_values]
                    X2_feature = [sample[i] for sample in pos_feature_values]
                    X1_feature = resample(X1_feature, replace=True, n_samples=len(X2_feature),
                                          random_state=RANDOM_STATE)

                    # 进行 t 检验
                    t_statistic, p_value = ttest_ind(X1_feature, X2_feature)
                    # 计算 Cohen's d
                    effect_size = cohen_d(X1_feature, X2_feature)

                    # 设置显著性水平（通常为 0.05）
                    alpha = 0.05

                    # 判断是否显著
                    if effect_size > 0.5:
                        significant_features_count += 1
                if significant_features_count > 10 and '@' in project_name:
                    # 打印显著的特征个数
                    print(project_name, end=',')

                if '@' not in project_name:
                    for value in pos_feature_values:
                        case_study_data_pos.append(value)
                    for value in neg_feature_values:
                        case_study_data_neg.append(value)
            print()
            neg_std = np.mean(case_study_data_neg, axis=0)
            pos_std = np.mean(case_study_data_pos, axis=0)
            case_study_difference = np.abs(neg_std - pos_std)
            # 绘制条形图
            # plt.bar(features, case_study_difference, color='skyblue')
            # plt.xlabel('Features')
            # plt.ylabel('Absolute Difference in Standard Deviation')
            # plt.title('Difference in Standard Deviation between arr1 and arr2')
            # plt.show()

            # 假设你有一个有序特征 X 和目标变量 y
            # 以 X 的每个分量逐个进行 t 检验
            for i in range(len(features)):
                X_feature_pos = [sample[i] for sample in case_study_data_pos]
                X_feature_neg = [sample[i] for sample in case_study_data_neg]

                # rearrange data structure
                X_feature_pos = resample(X_feature_pos, replace=True, n_samples=len(X_feature_neg),
                                         random_state=RANDOM_STATE)
                # 计算 Cohen's d
                effect_size = cohen_d(X_feature_pos, X_feature_neg)
                # 进行 t 检验
                t_statistic, p_value = ttest_ind(X_feature_pos, X_feature_neg)

                # 设置显著性水平（通常为 0.05）
                alpha = 0.05

                # 判断是否显著
                if effect_size > 0.15:
                    print(
                        f"Feature: {features[i]}, t-statistic: {t_statistic}, p-value: {p_value}, effect size: {effect_size}")

            exit(0)

        # 构建所有可能的特征子集
        all_feature_subsets = list(chain.from_iterable(combinations(features, r) for r in range(len(features), 0, -1)))
        # 初始化字典用于存储性能指标
        performance_metrics = {}
        # 创建决策树模型
        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        logging.info(f"all_feature_subsets: {len(all_feature_subsets)}")
        # 定义交叉验证折叠
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        i = 1  #
        for subset in all_feature_subsets[i - 1:-1]:
            print("index:", i, "subset:", subset, "subset size:", len(subset))
            i += 1
            X_subset = X[:, [features.index(feature) for feature in subset]]

            # 初始化 SMOTE == 对特征的顺序有影响的
            # sampler = SMOTE(random_state=42)

            # ADASYN（Adaptive Synthetic Sampling） 对顺序不敏感
            sampler = ADASYN(sampling_strategy='auto', random_state=RANDOM_STATE)
            precisions = []
            recalls = []
            # 交叉验证中的训练过程
            for train_index, test_index in cv.split(X_subset, y):
                X_train, X_test = X_subset[train_index], X_subset[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # 在训练集上应用 ADASYN
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

                # 训练模型
                clf.fit(X_train_resampled, y_train_resampled)

                # 在测试集上评估性能，不应用 SMOTE
                y_pred = clf.predict(X_test)
                # 统计 1的个数
                # print(Counter(y_test))

                # 计算 precision 和 recall
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = 2 * precision * recall / (precision + recall)
                precisions.append(precision)
                recalls.append(recall)
                if f1 < 0.30:
                    break

            # 计算平均 precision 和 recall
            precision = round(np.mean(precisions) * 100, 2)
            recall = round(np.mean(recalls) * 100, 2)
            f1 = round(2 * precision * recall / (precision + recall), 2)
            if f1 < 51.9:
                print(f"Precision: {round(precision, 2)}, Recall: {recall}, f1:{f1}")
                continue
            # 存储性能指标
            performance_metrics[str(subset)] = {'precision': precision, 'recall': recall}
            logging.info(
                f"Precision: {precision}, Recall: {recall}, f1: {f1}, Subset: {subset}, Subset size: {len(subset)}")
        # 打印所有性能指标
        logging.info(f"All Performance Metrics:{performance_metrics}")

        exit(0)

    # 定义十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    projects_map = {}
    projects_metrics = {}
    for feature_index in range(X.shape[1]):
        accuracies = []
        precisions = []
        recalls = []
        tp_and_fp = []
        tps = []
        X_scaled_data = X
        cnt_lg = 0
        cnt_sm = 0

        # 创建剔除当前特征的新特征矩阵
        features_without_feature = np.delete(features, feature_index, axis=0)
        X_scaled_data = np.delete(X, feature_index, axis=1)

        if feature_elimination_experiment_enable == False:
            X_scaled_data = X
            features_without_feature = features
        # 进行交叉验证
        for fold, (train_index, test_index) in enumerate(kf.split(X_scaled_data)):

            # 划分训练集和测试集 copy 不改写原有数据
            X_train, X_test = X_scaled_data[train_index].copy(), X_scaled_data[test_index].copy()
            y_train, y_test = y[train_index].copy(), y[test_index].copy()
            # 训练时用ValExtractor的数据
            for index in range(0, len(X_train)):
                name = index_to_data_map[train_index[index]]
                refactored_name = name.split('_')[0] + "_" + name.split('_')[1]
                if name.endswith('_1'):
                    new_data = pos_parser.data[refactored_name + ".json"].copy()
                    if refactored_name in positive_valExtractor_map:
                        index_list = positive_valExtractor_map[refactored_name]
                    else:
                        index_list = [i for i in range(0, len(new_data['expressionList']))]
                    # 　去除可以提取的数据
                    if max(index_list) < len(new_data['expressionList']):
                        new_data['occurrences'] = len(index_list)
                        new_data['expressionList'] = [new_data['expressionList'][index] for index in index_list]
                    # pos_maps
                    new_features = pos_parser.compute_features(new_data, features_without_feature)
                else:
                    new_data = neg_parser.data[refactored_name + ".json"].copy()
                    if refactored_name in negative_valExtractor_map:
                        index_list = negative_valExtractor_map[refactored_name]
                    else:
                        index_list = [i for i in range(len(new_data['expressionList']))]
                    # 　去除可以提取的数据
                    if index_list == []:
                        new_data['occurrences'] = 0
                    elif max(index_list) < len(new_data['expressionList']):  # index_list!=[]  拒绝提取
                        new_data['occurrences'] = len(index_list)
                        new_data['expressionList'] = [new_data['expressionList'][index] for index in index_list]
                    new_features = neg_parser.compute_features(new_data, features_without_feature)
                # if np.any(new_features != X_train[index]):
                #     pass
                X_train[index] = new_features

            if model_name != 'RandomForest' and model_name != 'DecisionTree' and model_name != 'NaiveBayes':
                # 对训练集进行标准化
                scaler = StandardScaler()
                scaler.fit(X_train)  # 标准化训练集 对象
                X_train = scaler.transform(X_train)  # 转换训练集
                X_test = scaler.transform(X_test)  # 转换测试集

            # ADASYN（Adaptive Synthetic Sampling） 对顺序不敏感
            sampler = ADASYN(sampling_strategy='auto', random_state=RANDOM_STATE)

            # 进行过采样
            X_train, y_train = sampler.fit_resample(X_train, y_train)


            # 通过对多数类样本进行有放回或无放回地随机采样来选择部分多数类样本。
            # cc = RandomUnderSampler(random_state=42)
            # X_train, y_train = cc.fit_resample(X_train, y_train)

            # 对训练集进行标准化
            # scaler = StandardScaler()  # 标准化转换
            # scaler.fit(X_train)  # 标准化训练集 对象
            # X_train_norm = scaler.transform(X_train)  # 转换训练集
            # X_test_norm = scaler.transform(X_test)  # 转换测试集

            X_train_norm = X_train
            X_test_norm = X_test
            # 训练模型
            clf.fit(X_train_norm, y_train)

            # y_predict = clf.predict(test_data)

            X_test_copy = X_test.copy()
            # 评估模型
            # score = clf.score(X_test, y_test)

            # for index in range(0, len(X_test)):
            #     if 'occurrences' in features:
            #         # 判断对象是否为数组
            #         if isinstance(val_extractor_data[index_to_data_map[test_index[index]]], list):
            #             # print(val_extractor_data[index_to_data_map[test_index[index]]])
            #             X_test_copy[index][0] = val_extractor_data[index_to_data_map[test_index[index]]][1]
            #         else:
            #             X_test_copy[index][0] = val_extractor_data[index_to_data_map[test_index[index]]]

            # X_test_norm_copy = scaler.transform(X_test_copy)  # 转换测试集
            y_predict = clf.predict(X_test_norm)

            for index in range(0, len(X_test)):
                # if y_predict[index] == 1:
                name = index_to_data_map[test_index[index]]
                refactored_name = name.split('_')[0] + "_" + name.split('_')[1]
                if name.endswith('_1'):
                    new_data = pos_parser.data[refactored_name + ".json"].copy()
                    if refactored_name in positive_valExtractor_map:
                        index_list = positive_valExtractor_map[refactored_name]
                    else:
                        index_list = [i for i in range(len(new_data['expressionList']))]
                    # 　去除可以提取的数据
                    if max(index_list) < len(new_data['expressionList']):
                        new_data['occurrences'] = len(index_list)
                        new_data['expressionList'] = [new_data['expressionList'][index] for index in index_list]
                    # pos_maps
                    new_features = pos_parser.compute_features(new_data, features_without_feature)
                else:
                    new_data = neg_parser.data[refactored_name + ".json"].copy()
                    if refactored_name in negative_valExtractor_map:
                        index_list = negative_valExtractor_map[refactored_name]
                    else:
                        index_list = [i for i in range(len(new_data['expressionList']))]
                    # 　去除可以提取的数据
                    if index_list == []:
                        new_data['occurrences'] = 0
                    elif max(index_list) < len(new_data['expressionList']):
                        new_data['occurrences'] = len(index_list)
                        new_data['expressionList'] = [new_data['expressionList'][index] for index in index_list]
                    new_features = neg_parser.compute_features(new_data, features_without_feature)
                # print(data)
                # tmp = X_test_copy[index].copy()  # 判断对象是否为数组
                # if isinstance(val_extractor_data[index_to_data_map[test_index[index]]], list):
                #     tmp[0] = val_extractor_data[index_to_data_map[test_index[index]]][2]
                # tmp_norm = scaler.transform([tmp])  # 重新转换测试集
                y_predict[index] = clf.predict([new_features])[0]
                # if y_predict[index] == 0 and y_test[index] == 0:
                #     print(name)
                    # print(
                    #     index_to_data_map[test_index[index]] + "," + "valextractor" + ":" + str(
                    #         tmp[0]) + "," + "original:" +  str(X_test_copy[index][0]))
                pass

            # 计算 precision 和 recall
            tp, fp, tn, fn = 0, 0, 0, 0
            for i in range(len(y_predict)):
                project_name = index_to_data_map[test_index[i]].split('_')[0]
                if y_test[i] == y_predict[i] and y_predict[i] == 1:
                    tp += 1
                    status = 'tp'
                elif y_test[i] == y_predict[i] and y_predict[i] == 0:
                    tn += 1
                    status = 'tn'
                elif y_test[i] != y_predict[i] and y_predict[i] == 1:
                    fp += 1
                    status = 'fp'
                    if '@' not in index_to_data_map[test_index[i]]:
                        # print("fp: " + index_to_data_map[test_index[i]], "features:" +
                        #       str(X_test_norm[i]))
                        pass
                elif y_test[i] != y_predict[i] and y_predict[i] == 0:
                    fn += 1
                    status = 'fn'
                    # print("fn: " + index_to_data_map[test_index[i]] + "," + "occurences" + ":" + str(
                    #     X_test_copy[i][0]) + "," + "features:" +
                    #       str(X_test_norm[i]))
                if project_name in projects_map:
                    projects_map[project_name][status] = projects_map[project_name][status] + 1
                else:
                    projects_map[project_name] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

            accuracy = 0 if tp + tn + fp + fn == 0 else (tp + tn) * 1.0 / (tp + tn + fp + fn)
            precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
            recall = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)

            # accuracy = accuracy_score(y_test, y_predic )
            # precision = precision_score(y_test, y_predict )
            # recall = recall_score(y_test, y_predict )

            # print(f"this fold Precision: {precision}, Recall: {recall}")
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
            print(f'f1: {round(2 * precision * recall / (precision + recall) * 100, 2) if precision + recall != 0 else 0}　')
            # print(f"pos acc {round(tp/(tp+ fn)* 100, 2)}:")
            # print(f'{tp + fp} {tp}')

            print('')
        if feature_elimination_experiment_enable:
            logging.info(
                f' model is {model_name}, exclude {features[feature_index]}, here are final results: ')  # {clf.get_depth()}
        else:
            logging.info(
                f' model is {model_name}, considering {features_without_feature}, here are final results: ')  # {clf.get_depth()}
        a = round(np.mean(accuracies) * 1, 2)
        p = round(np.mean(precisions) * 1, 2)
        r = round(np.mean(recalls) * 1, 2)
        f1 = round(2 * p * r / (p + r), 2)
        logging.info(f'precision:{p}')
        logging.info(f'recall:{r}')
        logging.info(f'accuracy:{a}')
        logging.info(f'f1:{f1}')
        logging.info(f'推荐总数{sum(tp_and_fp)}, 对的个数{sum(tps)}')
        if feature_elimination_experiment_enable == False:
            excluded_project = []

            # 每条数据是对的还是错的，目的是看哪些项目效果好
            for project in projects_map.items():
                # 按照项目计算precision recall
                # tp, fp, tn, fn = project['tp'], project['fp'], project['tn'], project['fn']
                tp, fp, tn, fn = project[1]['tp'], project[1]['fp'], project[1]['tn'], project[1]['fn']
                precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
                recall = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)
                if '@' in project[0]:
                    print(f"project {project} Precision: {precision}, Recall: {recall}")
                else:
                    logging.info(f"project {project} Precision: {precision}, Recall: {recall}")
                if (precision < 0.5) and '@' in project[0]:
                    excluded_project.append(project[0])
            print(excluded_project)
            break

    if model_name == 'DecisionTree':
        # 实例化SMOTE对象
        smote = SMOTE(random_state=RANDOM_STATE)
        # 进行过采样
        X_smote, y_smote = smote.fit_resample(X, y)
        clf.fit(X_smote, y_smote)
        y_pred = clf.predict(X)
        y_test = y
        tree_rules = export_text(clf, feature_names=features)
        # 指定图幅大小
        # plt.figure(figsize=(30, 35), dpi=200)
        # _ = tree.plot_tree(clf, fontsize=10, feature_names=features, filled=True, rounded=True, class_names=['0', '1'])
        print("plotting decision tree...")
        # ax = pybaobabdt.drawTree(clf, size=10, dpi=300, features=features)  # 可视化主函数pybaobabdt.drawTree
        # plt.show()
        # plt.savefig('./resource/decision_tree.png', format='png')

        # # 获取叶子节点的索引
        # leaf_indices = clf.apply(X)
        #
        # # 初始化一个空字典来保存每个叶子节点的 precision
        # leaf_precisions = {}

        # # 指定叶子节点的索引
        # leaf_index = 4
        #
        # # 获取到达叶子节点的条件
        # condition = get_node_condition(clf.tree_, leaf_index)
        # print(f"到达叶子节点 {leaf_index} 的条件为: {condition}")

        # # 遍历每个叶子节点
        # for leaf_index in np.unique(leaf_indices):
        #     # 获取当前叶子节点的预测结果和真实标签
        #     leaf_y_pred = y_pred[leaf_indices == leaf_index]
        #     leaf_y_true = y_test[leaf_indices == leaf_index]
        #
        #     tp, fp, tn, fn = 0, 0, 0, 0
        #     for i in range(len(leaf_y_pred)):
        #         if leaf_y_true[i] == leaf_y_pred[i] and leaf_y_pred[i] == 1:
        #             tp += 1
        #         elif leaf_y_true[i] == leaf_y_pred[i] and leaf_y_pred[i] == 0:
        #             tn += 1
        #         elif leaf_y_true[i] != leaf_y_pred[i] and leaf_y_pred[i] == 1:
        #             fp += 1
        #         elif leaf_y_true[i] != leaf_y_pred[i] and leaf_y_pred[i] == 0:
        #             fn += 1
        #
        #     if tp + fp == 0:
        #         continue
        #
        #     precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
        #     pstr = f"推荐{tp + fp}, 对了{tp}, 正确率{precision}"  # precision_score(leaf_y_true, leaf_y_pred)
        #
        #     # 将 precision 存储到字典中
        #     leaf_precisions[leaf_index] = pstr
        #
        # # 打印每个叶子节点的 precision
        # for leaf_index, precision in leaf_precisions.items():
        #     print(f"Leaf {leaf_index}: Precision = {precision}")
        # # print(X_test)
        # # print(clf.predict([[1, 15, 0, 0]]))
        exit(0)
        neg_case_study_parser = JsonParser(
            "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\negative\\", 0)
        pos_case_study_parser = JsonParser(
            "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\casestudy\\positive\\", 0)

        # 读取特征数据
        neg_case_study_maps = neg_case_study_parser.get_value(features)
        pos_case_study_maps = pos_case_study_parser.get_value(features)

        # map总的数据到每条数据id的映射关系
        case_study_index_to_data_map = {}
        neg_case_study_value_list = []
        pos_case_study_value_list = []
        for key in neg_case_study_maps.keys():
            case_study_index_to_data_map[len(case_study_index_to_data_map)] = key
            neg_case_study_value_list.append(neg_case_study_maps[key])
        for key in pos_case_study_maps.keys():
            case_study_index_to_data_map[len(case_study_index_to_data_map)] = key
            pos_case_study_value_list.append(pos_case_study_maps[key])

        neg_case_study_values = np.array(neg_case_study_value_list)[:len(neg_case_study_value_list)]
        pos_case_study_values = np.array(pos_case_study_value_list)[:len(pos_case_study_value_list)]

        case_study_X = np.concatenate((neg_case_study_values, pos_case_study_values))

        logging.info(f"Sample number: {len(case_study_X)}")
        case_study_y = np.concatenate(
            (np.zeros(len(neg_case_study_values)), np.ones(len(pos_case_study_values))))
        case_study_y_pred = clf.predict(case_study_X)

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(case_study_y_pred)):
            if case_study_y[i] == case_study_y_pred[i] and case_study_y_pred[i] == 1:
                tp += 1
            elif case_study_y[i] == case_study_y_pred[i] and case_study_y_pred[i] == 0:
                tn += 1
            elif case_study_y[i] != case_study_y_pred[i] and case_study_y_pred[i] == 1:
                fp += 1
            elif case_study_y[i] != case_study_y_pred[i] and case_study_y_pred[i] == 0:
                fn += 1
                # print("fp: " + case_study_index_to_data_map[i] + "," + "features:" + str(case_study_X[i]))
                # print("fn: " + index_to_data_map[test_index[i]] + "," + "occurences" + ":" + str(
                #     X_test_copy[i][0]) + "," + "features:" +
                #       str(X_test_norm[i]))
        accuracy = 0 if tp + tn + fp + fn == 0 else (tp + tn) * 1.0 / (tp + tn + fp + fn)
        precision = 0 if tp + fp == 0 else tp * 1.0 / (tp + fp)
        recall = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)

        precision = precision_score(case_study_y, case_study_y_pred, average='weighted')
        recall = recall_score(case_study_y, case_study_y_pred, average='weighted')

        recall_positive = 0 if tp + fn == 0 else tp * 1.0 / (tp + fn)
        accuracies.append(round(accuracy * 100, 2))
        precisions.append(round(precision * 100, 2))
        recalls.append(round(recall * 100, 2))
        tps.append(tp)
        tp_and_fp.append(tp + fp)

        logging.info('')
        logging.info(f"validation:")
        logging.info(f'precision: {round(precision * 100, 2)}')
        logging.info(f'recall: {round(recall * 100, 2)}')
        logging.info(f'accuracy: {round(accuracy * 100, 2)}')
        logging.info(f'f1: {round(2 * precision * recall / (precision + recall) * 100, 2)}')
        # print(f"pos acc {round(tp/(tp+ fn)* 100, 2)}:")
        # print(f'{tp + fp} {tp}')
        logging.info('')

        # # 获取特征重要性
        # feature_importance = clf.feature_importances_
        # # 打印特征重要性
        # print("Feature Importance:")
        # for i, importance in enumerate(feature_importance):
        #     print(f"Feature {i + 1}: {importance}")
