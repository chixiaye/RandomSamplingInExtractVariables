from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 创建训练数据集
X = np.array([(5, 'A'), (10, 'B'), (15, 'C'), (20, 'A')])
y = np.array([0, 1, 1, 0])

# 将类型转换为数值向量
enc = OneHotEncoder()
X_enc = enc.fit_transform(X[:, 1].reshape(-1, 1)).toarray()

# 合并长度和类型的数值向量
X_final = np.column_stack((X[:, 0], X_enc))

# 创建支持向量机分类器
clf = SVC(kernel='linear')
clf.fit(X_final, y)

# 使用分类器进行预测
X_test = np.array([(8, 'C'), (18, 'B')])
X_test_enc = enc.transform(X_test[:, 1].reshape(-1, 1)).toarray()
X_test_final = np.column_stack((X_test[:, 0], X_test_enc))
y_pred = clf.predict(X_test_final)
print(y_pred)
