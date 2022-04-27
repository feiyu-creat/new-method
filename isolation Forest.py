# 加载模型所需要的的包
import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from pandas import ExcelWriter
sheetname1=['4.75档','9.5档','13.2档','16档',]

# df=data[['Volume','Area']]


for i in range (len(sheetname1)):
    sheetname=sheetname1[i]
    # 构造一个数据集，只包含一列数据，假如都是月薪数据，有些可能是错的
    data = pd.read_excel('data/4.75-yes-no.xlsx', sheet_name=sheetname)
    df = data.drop(['plyIndex', '分档', 'target', 'Pmoments_i2', 'Pmoments_i4'], axis=1)  # 删除这些列的信息
    df1=pd.DataFrame()
    #构建模型 ,n_estimators=100 ,构建100颗树
    model = IsolationForest(n_estimators=200,
                          max_samples='auto',
                          contamination=float(0.1),

                          max_features=1.0)
    # 训练模型
    model.fit(df)

    # 预测 decision_function 可以得出 异常评分
    data['scores']  = model.decision_function(df)

    #  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
    data['anomaly'] = model.predict(df)
    # data['if']=data['anomaly']
    path ='data'+'/'+'isofor'+ sheetname1[i] +'.xlsx'


    with ExcelWriter(path) as writer:
        data.to_excel(writer, sheet_name=sheetname1[i])



print('运行结束！！')