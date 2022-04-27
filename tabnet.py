import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

gpus = [0]   #使用哪几个GPU进行训练，这里选择0号GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using {}".format(DEVICE))


# 假定读入二分类表格数据data，标签为data_label，需保证data全为数值型特征，若存在字符型，需编码为数值型
file1="data/4.75-yes-no.xlsx"
test_size=0.3
random_state=20


aggregate_data = pd.read_excel(file1,
                               sheet_name='4.75-del-9.5-500',
                               # encoding='gb18030'
                               )
# aggregate_data = pd.read_csv(file1,encoding='gb18030')
# print(aggregate_data.isnull())
aggregate_feature = aggregate_data.drop(['plyIndex','Pmoments_i2','Pmoments_i4','分档','target'], axis=1)#删除这些列的信息
# aggregate_feature = aggregate_data.drop(['plyindex','分档','target','Pmoments_i2','Pmoments_i4'], axis=1)  # 删除这些列的信息
aggregate_target = aggregate_data['target']

# aggregate_feature = aggregate_data.drop(['序号', '质量', '人工测量：长', '人工测量：高', '分档', '宽1', '宽2', '宽3', '人工测量：宽','边1','边2','边3','备注','Pinner_height', 'Pmoments_psi3','Pmoments_psi4','Pmax_diameter','Prect2_len1','Pinner_radius'], axis=1)  # 删除这些列的信息
# aggregate_target = aggregate_data['分档']

X = aggregate_feature#特征
y = aggregate_target#目标


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
X_test_v, val_x, y_test_v, val_y = train_test_split(X_test, y_test, test_size=test_size, random_state=random_state)

X_train = X_train.fillna(-1).values
val_x = val_x.fillna(-1).values
X_test = X_test.fillna(-1).values
y_train = y_train.values
val_y = val_y.values
y_test = y_test.values

tabnet_params = dict(
    n_d = 8,  # 可以理解为用来决定输出的隐藏层神经元个数。n_d越大，拟合能力越强，也容易过拟合
    n_a = 8,   # 可以理解为用来决定下一决策步特征选择的隐藏层神经元个数
    n_steps = 3, # 决策步的个数。可理解为决策树中分裂结点的次数
    gamma = 1.3,  # 决定历史所用特征在当前决策步的特征选择阶段的权重，gamma=1时，表示每个特征在所有决策步中至多仅出现1次
    lambda_sparse = 1e-3,  # 稀疏正则项权重，用来对特征选择阶段的特征稀疏性添加约束,越大则特征选择越稀疏
    optimizer_fn = torch.optim.Adam, # 优化器
    optimizer_params = dict(lr = 0.02, weight_decay = 1e-5),
    momentum = 0.03,
    mask_type = "entmax",
    seed = 0,
    scheduler_params = {"gamma": 0.95, "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    epsilon=1e-15,
    device_name = DEVICE

)

clf = TabNetClassifier(**tabnet_params)
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (val_x, val_y)],
    eval_name=['train', 'valid'],
    eval_metric=['logloss', 'accuracy'],
    max_epochs=200,  # 最大迭代次数
    patience=50,    # 在验证集上早停次数，
    batch_size=128, # BN作用在的输入特征batch
    virtual_batch_size=16,  # 除了作用于模型输入特征的第一层BN外，都是用的是ghost BN。
    num_workers=0,
    drop_last=False,


)



print('文件名：',file1,'\n'
      'test_size:',test_size,'\n'
      '参数：',tabnet_params
                    )
# y_pred = clf.predict_proba(X_test)[:,1]
y_pred = clf.predict(X_test)
# y_pred = clf.prepare_target(y_test)
# print(y_pred,y_test)
print('预测精度：',accuracy_score(y_test,y_pred))

print('精度报告')
print(metrics.classification_report(y_test, y_pred,digits=4))
his=clf.history

plt.plot(clf.history['lr'])
plt.legend(["lr"],loc="best")
plt.show()
plt.plot(clf.history['loss'])
plt.legend(["loss"],loc="best")
plt.show()

plt.plot(clf.history['train_accuracy'])
plt.plot(clf.history['valid_accuracy'])
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(["train-acc","val-acc"],loc="best")
plt.show()

plt.plot(clf.history['train_logloss'])
plt.plot(clf.history['valid_logloss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(["train-loss","val-loss"],loc="best")
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
cm_tab=confusion_matrix(y_test,y_pred)

con_mat_norm = cm_tab.astype('float') / cm_tab.sum(axis=1)[:, np.newaxis]##牛
print(cm_tab,type(cm_tab),con_mat_norm)
sns.heatmap(con_mat_norm,annot=True,cmap="RdBu_r",fmt=".2f",cbar=False, annot_kws={"size": 18})
plt.show()
sns.heatmap(cm_tab,annot=True,cmap="RdBu_r",fmt="d",cbar=False, annot_kws={"size": 18})
plt.show()