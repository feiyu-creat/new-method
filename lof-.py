from pyod.models.lof import LOF
from pyod.utils.data import generate_data
from pyod.utils import precision_n_scores
import numpy as np
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

# X_train, y_train, X_test, y_test = \
#         generate_data(n_train=200,
#                       n_test=100,
#                       n_features=5,
#                       contamination=0.1,
#                       random_state=3)


file1="data/4.75-yes-no.xlsx"
test_size=0.3
random_state=20
max_features = 55

aggregate_data = pd.read_excel(file1,sheet_name='4.75-del-9.5-500')
# print( aggregate_data.isnull())

# aggregate_feature = aggregate_data.drop(['序号', '质量','人工测量：长','人工测量：高', '分档','宽1','宽2','宽3','人工测量：宽','边1','边2','边3','备注'], axis=1)#删除这些列的信息
aggregate_feature = aggregate_data.drop(['plyIndex','quality','分档','target','Pmoments_i2','Pmoments_i4'], axis=1)  # 删除这些列的信息
aggregate_target = aggregate_data['target']
X = aggregate_feature#特征
y = aggregate_target#目标
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=random_state)


X_train = X_train * np.random.uniform(0, 1, size=X_train.shape)
X_test = X_test * np.random.uniform(0,1, size=X_test.shape)
clf_name = 'LOF'
clf = LOF()
clf.fit(X_train)

test_scores = clf.decision_function(X_test)

# roc = round(roc_auc_score(y_test, test_scores), ndigits=6)
# prn = round(precision_n_scores(y_test, test_scores), ndigits=6)

# print(f'{clf_name} ROC:{roc}, precision @ rank n:{prn}')
import matplotlib.pyplot as plt

detector = LOF()
scores = detector.fit(X_train).decision_function(X_test)

sns.histplot(scores[y_test==0], label="inlier scores",color='red')
sns.histplot(scores[y_test==1], label="outlier scores",color="blue")
plt.legend()
plt.xlabel("Outlier score")
plt.show()