"""
Xgboost训练、精确度测试、贝叶斯优化、代码
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score as CVS
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from yellowbrick.classifier import confusion_matrix


"""
主函数是训练代码
"""
if __name__ == '__main__':

    file1="data/4.75-yes-no.xlsx"
    test_size=0.3
    random_state=20
    max_features = 55

    aggregate_data = pd.read_excel(file1,
                                   sheet_name='4.75-del-9.5-500',
                                 # encoding='gb18030'
                                   )
    # print( aggregate_data.isnull())

    # aggregate_feature = aggregate_data.drop(['序号', '质量','人工测量：长','人工测量：高', '分档','宽1','宽2','宽3','人工测量：宽','边1','边2','边3','备注'], axis=1)#删除这些列的信息
    aggregate_feature = aggregate_data.drop(['plyIndex','分档','target','Pmoments_i2','Pmoments_i4'], axis=1)  # 删除这些列的信息
    aggregate_target = aggregate_data['target']
    # aggregate_feature = aggregate_data.drop(
    #     ['序号', '质量', '人工测量：长', '人工测量：高', '分档', '宽1', '宽2', '宽3', '人工测量：宽', '边1', '边2', '边3', '备注', 'Pinner_height',
    #      'Pmoments_psi3', 'Pmoments_psi4', 'Pmax_diameter', 'Prect2_len1', 'Pinner_radius'], axis=1)  # 删除这些列的信息
    # aggregate_target = aggregate_data['分档']

    X = aggregate_feature#特征
    y = aggregate_target#目标
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  # 测试集占总样本的百分比0.3，训练集0.7,random_state=20


    """
    特征选择
    """


    # params = {'learning_rate': 0.09, 'n_estimators': 200, 'max_depth': 4, 'min_child_weight': 3,
    #           'subsample': 0.9, 'colsample_bytree': 0.2, 'gamma': 0, 'reg_alpha': 1, 'reg_lambda': 0.9}
    params = {'colsample_bytree': 0.790039921542529, 'gamma': 0.4807147736689332, 'learning_rate': 0.11008143934442158,
              'max_depth': 4, 'min_child_weight': 0.004584015892690196, 'n_estimators': 161, 'reg_alpha': 1.9301644285406032,
              'reg_lambda': 0.1589419009086196}

    model1 =XGBClassifier(**params,tree_method='gpu_hist',use_label_encoder=False)
    model1.fit(X, y)

    sele_model=SelectFromModel(model1,max_features=max_features,prefit=True,threshold=-np.inf)
    X_new=sele_model.transform(X)
    # print(X_new.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X_new, y,test_size=test_size, random_state=random_state)





    """
    贝叶斯优化参数up
    """
    def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight,  colsample_bytree, reg_alpha, gamma,reg_lambda):
        # print(max_depth, learning_rate, n_estimators, min_child_weight,  colsample_bytree, reg_alpha, gamma)
        res = cross_val_score(
            estimator=XGBClassifier(max_depth=int(max_depth),
                                    learning_rate=learning_rate,
                                    n_estimators=int(n_estimators),
                                    min_child_weight=min_child_weight,
                                    # subsample=max(min(subsample, 1), 0),
                                    colsample_bytree=max(min(colsample_bytree, 1), 0),
                                    reg_alpha=max(reg_alpha, 0),
                                    gamma=gamma,
                                    objective='reg:squarederror',
                                    tree_method='gpu_hist',
                                    reg_lambda=max(reg_lambda, 0),
                                    use_label_encoder=False,
                                    eval_metric=['logloss', 'auc', 'error']
                                    ), X=X_train, y=y_train,  cv=3
        ).mean()
        # print(res)
        return res
    xgb_bo = BayesianOptimization(
        xgb_cv,
        pbounds={'max_depth': (2,7),
                 'learning_rate': (0.05, 0.15),
                 'n_estimators': (90,200),
                 'min_child_weight': (0, 20),
                 # 'subsample': (0.001, 1),
                 'colsample_bytree': (0.01, 1),
                 'reg_alpha': (0.001, 20),
                 'gamma': (0, 10),
                 'reg_lambda':(0,1)
                 })

    xgb_bo.maximize(init_points=25,
                    n_iter=50)  # n_iter去定义我们执行贝叶斯优化的步数，步数越多，越容易找到一个更好的最大值，但是也会消耗时间，需要手动指定，init_points去定义需要执行多少步随机搜索，随机搜索可以提升探索空间的范围
    # 总共训练20+3轮
    a=xgb_bo.max
    a['params']['max_depth'] = int(a['params']['max_depth'])
    a['params']['n_estimators'] = int(a['params']['n_estimators'])
    print('最佳参数：', a)
    """
    贝叶斯参数优化down
    """


    # cv_params = {'learning_rate': [0.1, 0.2, 0.3, 0.4]}
    # other_params = {'learning_rate': 0.05, 'n_estimators':146, 'max_depth': 4, 'min_child_weight': 3,#初始参数
    #                 'subsample': 0.9, 'colsample_bytree': 0.2, 'gamma': 0, 'reg_alpha': 1, 'reg_lambda': 0.9,tree_method:'gpu_hist',use_label_encoder:False}#原来的参数-未经交叉验证调过的参数
    # params = {'learning_rate': 0.09, 'n_estimators': 200, 'max_depth': 4, 'min_child_weight': 3,
    #                 'subsample': 0.9, 'colsample_bytree': 0.2, 'gamma': 0, 'reg_alpha': 1, 'reg_lambda': 0.9}
    # params = {'learning_rate': 0.10191297577338768, 'n_estimators': 180, 'max_depth': 2, 'min_child_weight': 1.2022360150408784,
    #           'subsample': 0.9, 'colsample_bytree': 0.8194659922193182, 'gamma': 2.0600265936456252, 'reg_alpha': 0.3567235892087158, 'reg_lambda': 0.12670828017625624}##贝叶斯参数试验
    params=a['params']
    model = XGBClassifier(**params,tree_method='gpu_hist',use_label_encoder=False)#之前(r：0.09，n_est:200)

    model.fit(X_train, y_train)

    """
    可视化 
    """
    confusion_matrix(model,X_train,y_train,X_test,y_test,
                     # encoder={0:'13.2',1:'16',2:'19'},
                     percent=True,)



"""
交叉验证
# """
rs=[]
rs.append(CVS(model, X_train, y_train,scoring='accuracy', cv=5).mean())
print("##############################################################")
print('5折交叉验证：{}'.format(max(rs)) )


"""
试验log
"""
print('最佳参数：', a)
print("文件名,数据集参数:{}".format(file1))
print('测试集比例：',test_size,'随机数：',random_state)
print('本次试验参数：{}'.format(params))
print('特征选择个数：',max_features)




# pickle.dump(model, open("model/9.13-方案三.dat", "wb"))#模型保存

"""
训练好之后，不用再训练就可以用测试集进行测试精确度，需注释前面的主函数，/model/9.13-del-diff.dat
"""
# model = pickle.load(open("model/9.13-del-diff.dat", "rb"))
# aggregate_data = pd.read_csv("细分档方案二数据 .csv",encoding='gb18030')
# aggregate_feature = aggregate_data.drop(['序号', '质量','人工测量：长','人工测量：高', '分档','宽1','宽2','宽3','人工测量：宽'], axis=1)#删除这些列的信息
# aggregate_target = aggregate_data['分档']
# X = aggregate_feature#特征
# y = aggregate_target#目标
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)#测试集占总样本的百分比0.3，训练集0.7

"""
交叉验证进行参数选择
"""
# axisx = range(50,70,10)
# # axisx=[0.42,0.45,0.48,0.51,0.54,0.57,0.60,0.63,0.66,0.69,0.72,0.75,0.78,0.81,0.84,0.87,0.90]
# # axisx=[0.60,0.61,0.62,0.63,0.64,0.65,0.66]
# rs = []
# for i in axisx:
#     model = XGBClassifier(learning_rate=0.66, n_estimators=60, max_depth= 5, min_child_weight=2,subsample=1, colsample_bytree= 1, gamma= 0, reg_alpha= 1,
#                           reg_lambda= 0.9,tree_method='gpu_hist',use_label_encoder=False)
#     # reg = NGBClassifier(Dist=k_categorical(7),n_estimators=260, random_state=420,verbose_eval=20,col_sample=0.3,learning_rate=i)
#     rs.append(CVS(model, X_train, y_train,scoring='accuracy', cv=10).mean())  ###交叉验证的得分
# print('最大值索引：{}'.format(axisx[rs.index(max(rs))]),'最大值：{}'.format(max(rs)) )
# plt.figure(figsize=(20, 5))
# plt.plot(axisx, rs, c="red", label="XGB")
# plt.legend()
# print('结束')
# plt.show()





"""
验证代码
"""
b_output = model.predict(X_test)
predictions = [round(value) for value in b_output]
predictions1 = [round(value) for value in y_test]
#将预测结果与真实值写入csv文件
# pd_data_xtest = pd.DataFrame(b_output, columns=['pre'])
# pd_data_ytest= pd.DataFrame(predictions1, columns=['rea'])
# pd_data_xtest.to_csv('分档预测值.csv', index=False)
# pd_data_ytest.to_csv('分档真实值.csv', index=False)
print('精度报告')
print(metrics.classification_report(y_test, b_output))
# evaluate predictions
accuracy = accuracy_score(predictions1, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # print('平均得分score：',np.mean(score))
# plot_importance(model)  # 输出特征重要性排名
# pyplot.show()

"""
记录：
1.初次精度：84.93%
2.第二次精度：0.8446 'params': {'colsample_bytree': 0.8700917720720646, 'gamma': 3.317573376846923, 'learning_rate': 0.20366959046896368, 'max_depth': 1.557961977861327, 'min_child_weight': 0.18318832697309828, 'n_estimators': 299.0698716615637, 'reg_alpha': 0.6415391452911295, 'subsample': 0.9530735767626015}}
3.最佳参数： {'target': 0.8422524012386606, 'params': {'colsample_bytree': 0.9157574377359109, 'gamma': 3.4737518951722532, 'learning_rate': 0.7700021656224771, 'max_depth': 14.761016066093637, 'min_child_weight': 12.182462604547196, 'n_estimators': 499.71658460353984, 'reg_alpha': 0.24072411107975875, 'subsample': 0.8296693364847622}}
4.最佳参数： {'target': 0.8480791618160651, 'params': {'colsample_bytree': 0.656946474728726, 'gamma': 0.608241610386466, 'learning_rate': 0.3601275762653277, 'max_depth': 16.333554406995738, 'min_child_weight': 0.9807061582207721, 'n_estimators': 382.2546392032382, 'reg_alpha': 0.28814172025981405}}
5.10-50最佳参数： {'target': 0.8504074505238649, 'params': {'colsample_bytree': 0.6942536227938347, 'gamma': 0.1538121096893122, 'learning_rate': 0.3959147804893292, 'max_depth': 1.1923802265571015, 'min_child_weight': 0.8649526506256389, 'n_estimators': 499.268158832033, 'reg_alpha': 3.96074838449104}}
6.20-60最佳参数： {'target': 0.8506984866123399, 'params': {'colsample_bytree': 0.5143556064342273, 'gamma': 0.09530653892224893, 'learning_rate': 0.4329999859638225, 'max_depth': 17.660352810703042, 'min_child_weight': 1.5166006125001186, 'n_estimators': 434.88730548487536, 'reg_alpha': 0.33136708496812245}}
7.20-60最佳参数： {'target': 0.8515684431914798, 'params': {'colsample_bytree': 0.643458226441247, 'gamma': 4.526794257089074, 'learning_rate': 0.07753346923872406, 'max_depth': 19.83232881208894, 'min_child_weight': 1.413266187752218, 'n_estimators': 346.1430239031069, 'reg_alpha': 0.1395646741221514}}
8.20-60（全数据集）最佳参数： {'target': 0.8530267753201397, 'params': {'colsample_bytree': 0.694852887477539, 'gamma': 1.6248834131293095, 'learning_rate': 0.24713021919905453, 'max_depth': 4.390949285046318, 'min_child_weight': 2.183041405770554, 'n_estimators': 562.5889902431862, 'reg_alpha': 0.0685092789541649}}

重新修改数据集（前面的数据集有问题）
9.10-60最佳参数： {'target': 0.8512812700991841, 'params': {'colsample_bytree': 0.9452578889975385, 'gamma': 0.23324742579601376, 'learning_rate': 0.02582206890815194, 'max_depth': 16.182664699006956, 'min_child_weight': 0.27656043267240804, 'n_estimators': 316.48617260246743, 'reg_alpha': 1.5797789109154559}}

12.30
del26.5档-最佳参数： {'target': 0.8865708646254573, 'params': {'colsample_bytree': 0.5861839591592208, 'gamma': 0.22255964566848752, 'learning_rate': 0.0584374167754768, 'max_depth': 5.986672753899908, 'min_child_weight': 0.13322478429675932, 'n_estimators': 179.7738154092691, 'reg_alpha': 3.63226832450847, 'reg_lambda': 0.21451052635757162}}

22year
del26.5测试 - 平衡9.5档数据至400删去异常值.csv-最佳参数 {'target': 0.8929738058551617, 'params': {'colsample_bytree': 0.8443420393492385, 'gamma': 0.2355933430656365, 'learning_rate': 0.1195976324042266, 'max_depth': 5.37796721910583, 'min_child_weight': 0.6579898075397872, 'n_estimators': 150.04058939556862, 'reg_alpha': 3.3696827676665078, 'reg_lambda': 0.3097299204681885}}
"""





