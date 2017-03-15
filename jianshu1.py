# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filename = 'D:/kaggle/HR_comma_sep.csv'
data = pd.read_csv(filename)
# print(data.head())
data_connection = data.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(data_connection,vmax=1.0,annot=True,square=True,cmap="YlGnBu")
# plt.show()
# 因为共有十个变量，所以要/10
salary_low = data[data['salary'] == 'low']
salary_medium = data[data['salary'] == 'medium']
salary_high = data[data['salary'] == 'high']
# print('the number of salary_low:',salary_low.size/10)
# print('the number of salary_medium:',salary_medium.size/10)
# print('the number of salary_high:',salary_high.size/10)
# salary_low_mean=salary_low.mean()
# salary_low_std=salary_low.std()
# print(salary_low_mean.values.T)
# print(salary_low_std.values.T)
# salary_medium_mean=salary_medium.mean()
# salary_medium_std=salary_medium.std()
# print(salary_medium_mean)
# print(salary_medium_std)
# salary_high_mean=salary_high.mean()
# salary_high_std=salary_high.std()
# print(salary_high_mean)
# print(salary_high_std)
# plt.figure(figsize=(10, 10))
# sns.factorplot(x='sales', kind='count', data=data, col='salary', col_wrap=3)
# plt.show()
# sales_categaries = ['sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing',
#                     'RandD']
# sales_low_count = []
# sales_medium_count = []
# sales_high_count = []
# salary_low=data[data['salary']=='low']
# salary_medium=data[data['salary']=='medium']
# salary_high=data[data['salary']=='high']
# color=['c','m','r','b','yellow','green','gray','k','w']
# for i in sales_categaries:
#     sales_low_count.append(salary_low[salary_low['sales']==i].shape[0])
# for i in sales_categaries:
#     sales_medium_count.append(salary_medium[salary_medium['sales']==i].shape[0])
# for i in sales_categaries:
#     sales_high_count.append(salary_high[salary_high['sales']==i].shape[0])
# plt.subplot(131)
# plt.title('salary_low')
# plt.pie(sales_low_count,
#         labels=sales_categaries,
#         colors=color,
#         startangle=90,
#         shadow=True,
#         explode=(0.1,0,0,0,0,0,0,0,0,0),
#         autopct='%1.1f%%'
#         )
# plt.subplot(132)
# plt.title('salary_medium')
# plt.pie(sales_medium_count,
#         labels=sales_categaries,
#         colors=color,
#         startangle=90,
#         shadow=True,
#         explode=(0.1,0,0,0,0,0,0,0,0,0),
#         autopct='%1.1f%%'
#         )
# plt.subplot(133)
# plt.title('salary_high')
# plt.pie(sales_high_count,
#         labels=sales_categaries,
#         colors=color,
#         startangle=90,
#         shadow=True,
#         explode=(0.1,0,0,0,0,0,0,0,0,0),
#         autopct='%1.1f%%'
#         )
# plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

data_copy = pd.get_dummies(data)
data_copy_information = data_copy.drop(['left'], axis=1).values
data_copy_left = data_copy['left'].values
X_train, X_test, y_train, y_test = train_test_split(data_copy_information, data_copy_left, test_size=0.5)
# LR测试
# LR=LogisticRegression()
# LR.fit(X_train,y_train)
# print('LR\'s accuracy rate:',LR.score(X_test,y_test))
# 随机森林测试
rmf = RandomForestClassifier()
rmf.fit(X_train, y_train)
print('rmf\'s accuracy rate:', rmf.score(X_test, y_test))
# SVM测试
# svm_model=svm.SVC()
# svm_model.fit(X_train,y_train)
# print('svm\'s accuracy rate:',svm_model.score(X_test,y_test))
# 将特征重要性从小到大排序
indices = np.argsort(rmf.feature_importances_)[::-1]
# print(indices)
data_feature_names = []
data_feature_probability = []
for i in range(data_copy_information.shape[1]):
    # print('%d.feature %d %s(%f)'%(i+1,indices[i],data_copy.columns[indices[i]],rmf.feature_importances_[indices[i]]))
    data_feature_names.append(data_copy.columns[indices[i]])
    data_feature_probability.append(rmf.feature_importances_[indices[i]])
# plt.figure(figsize=(20,20))
# sns.set_style("whitegrid")
# sns.barplot(data_feature_names,data_feature_probability)
# plt.xlabel('feature')
# plt.ylabel('probability')
# plt.show()
now_stay = data[data['left'] == 0]
now_stay = pd.get_dummies(now_stay)
now_stay_information = now_stay.drop(['left'], axis=1).values
now_stay_label = now_stay['left'].values
predict = rmf.predict_proba(now_stay_information)
print(sum(predict[:, 1] == 1))
now_stay['maybe leave company']=predict[:,1]
print(now_stay[now_stay['maybe leave company']>=0.5].sort_values('maybe leave company',ascending=False))
outputfile='D:/kaggle/.output.csv'
now_stay[now_stay['maybe leave company']>=0.5].sort_values('maybe leave company',ascending=False).to_csv(outputfile)
