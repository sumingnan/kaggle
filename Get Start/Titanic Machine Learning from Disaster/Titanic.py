# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import cross_validation

absPath = os.path.abspath(os.path.dirname('train.csv'))
data_train = pd.read_csv(absPath + "/data/train.csv")

fig = plt.figure()
fig.set(alpha=0.2)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
'''
plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"获救情况（1为获救）")
plt.ylabel(u"人数")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.ylabel(u"人数")
plt.title(u"乘客等级公布")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布（1为获救）")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u"头等舱", u"2等舱", u"3等舱"), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()
'''
'''
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
plt.show()
'''
'''
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u'按性别看获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')
plt.show()
'''
'''
plt.title(u"根据舱等级和性别的获救情况")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()
'''
'''
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")

plt.show()
'''
def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    know_age = age_df[age_df.Age.notnull()].values
    unknow_age = age_df[age_df.Age.isnull()].values

    #用其他特征值预测年龄
    y = know_age[:,0]
    x = know_age[:,1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    predictedAge = rfr.predict(unknow_age[:,1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAge

    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
#dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Sex, dummies_Pclass], axis=1)
#child
df['Child'] = 0
df.loc[(df.Age < 12), 'Child'] = 1
#mother
df['Mother'] = 0
df.loc[((df.Parch > 1) & ('Mrs' in  df.Name)), 'Mother'] = 1
#fam size
df['Family_size'] = df['Parch'] + df['SibSp'] + 1

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(np.array(df['Age']).reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(np.array(df['Age']).reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(np.array(df['Fare']).reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1, 1), fare_scale_param)
print type(df)

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child|Mother|Family_size')
train_np = train_df.values
#训练
y = train_np[:,0]
x = train_np[:,1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(x, y)
print clf
'''
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child|Mother|Family_size')
X = all_data.values[:,1:]
Y = all_data.values[:,0]
print cross_validation.cross_val_score(clf, X, Y, cv=5)
'''
'''
split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child|Mother|Family_size')
clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf1.fit(train_df.values[:,1:], train_df.values[:,0])

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child|Mother|Family_size')
predictions = clf1.predict(cv_df.values[:,1:])

origin_data_train = pd.read_csv(absPath + '/data/Train.csv')
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'])]
print bad_cases
'''
#测试数据的处理


data_test = pd.read_csv(absPath + '/data/test.csv')
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values

X = null_age[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
#dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Sex, dummies_Pclass], axis=1)
#child
df_test['Child'] = 0
df_test.loc[(df_test.Age < 12), 'Child'] = 1
#mother
df_test['Mother'] = 0
df_test.loc[((df_test.Parch > 1) & ('Mrs' in  df_test.Name)), 'Mother'] = 1
#fam size
df_test['Family_size'] = df_test['Parch'] + df_test['SibSp'] + 1

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(np.array(df_test['Age']).reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(np.array(df_test['Fare']).reshape(-1, 1), fare_scale_param)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child|Mother|Family_size')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv(absPath + '/data/logistic_regression_prediction.csv', index=False)
