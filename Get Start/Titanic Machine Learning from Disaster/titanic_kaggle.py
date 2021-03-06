# -*- coding: utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os
from sklearn.model_selection import KFold

# Configure visualisations
#%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


def plot_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure(figsize=(16, 12))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=10, ax=ax)
        ax.set_title(
            'Skew: ' + str(round(float(df[var_name].skew()), )))  # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels([], visible=False)
        ax.set_yticklabels([], visible=False)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()


def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()


def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()


def plot_correlation_map(df):
    corr = titanic.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 12}
    )


def describe_more(df):
    var = [];
    l = [];
    t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by='Levels', inplace=True)
    return levels


def plot_variable_importance(X, y):
    tree = DecisionTreeClassifier(random_state=99)
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)


def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(
        model.feature_importances_,
        columns=['Importance'],
        index=X.columns
    )
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[: 10].plot(kind='barh')
    print (model.score(X, y))

# get titanic & test csv files as a DataFrame
absPath = os.path.abspath(os.path.dirname('train.csv'))
train = pd.read_csv(absPath + "/data/train.csv")
test = pd.read_csv(absPath + '/data/test.csv')

full = train.append(test, ignore_index= True)
titanic = full[:891]

del train, test
#print 'Datasets: ', 'full: ', full.shape, 'titanic:', titanic.shape

#print titanic.head()
#print titanic.describe()
#plot_correlation_map(titanic)
#plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
#plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )

# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( full.Sex == 'male', 1, 0), name='Sex')

# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies(full.Embarked, prefix='Embarked')
#print embarked.head()

# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies(full.Pclass, prefix='Pclass')
#print pclass.head()

# Create dataset
title = pd.DataFrame()
title['Title'] = full['Name'].map( lambda name:name.split(',')[1].split('.')[0].strip() )
# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
# we map each title
title['Title'] = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)
#print title.head()

imputed = pd.DataFrame()
scaler = preprocessing.StandardScaler()
# Fill missing values of Fare with the average of Fare (mean)
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())
def setMissAge(imputed, title, full):
    age_df = pd.DataFrame()
    age_df['Age'] = full['Age']
    age_df['Fare'] = imputed['Fare']
    age_df['Parch'] = full['Parch']
    age_df['SibSp'] = full['SibSp']
    age_df['Pclass'] = full['Pclass']
    age_df['Master'] = title['Master']
    age_df['Miss'] = title['Miss']
    age_df['Mr'] = title['Mr']
    age_df['Mrs'] = title['Mrs']
    age_df['Officer'] = title['Officer']
    age_df['Royalty'] = title['Royalty']
    #age_df['Title'] = title['Title']

    knowAge = age_df[age_df.Age.notnull()].values
    unknowAge = age_df[age_df.Age.isnull()].values

    y = knowAge[:,0]
    X = knowAge[:,1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictAges = rfr.predict(unknowAge[:,1::])
    age_df.loc[age_df.Age.isnull(), 'Age'] = predictAges
    #print age_df['Age']
    return age_df['Age']

age = setMissAge(imputed, title, full)

fare_scale_param = scaler.fit(np.array(imputed['Fare']).reshape(-1, 1))
imputed['Fare'] = scaler.fit_transform(np.array(imputed['Fare']).reshape(-1, 1), fare_scale_param)
# Fill missing values of Age with the average of Age (mean)
imputed['Age'] = age
imputed['Child'] = imputed['Age'].map(lambda s : 1 if s < 12 else 0)
#imputed['Age_Deg'] = imputed['Age'].map(lambda s : 1 if s < 8 else ( 2 if s < 18 else (3 if s < 29 else ( 4 if s < 41 else (5 if s < 66 else 6) ) ) ) )
imputed['Mother'] = 0
imputed.loc[((full.Parch > 1) & (str(full.Name).find('Mrs') > -1)), 'Mother'] = 1
#imputed = imputed.drop(['Age'], axis=1)
age_scale_param = scaler.fit(np.array(imputed['Age']).reshape(-1, 1))
imputed['Age'] = scaler.fit_transform(np.array(imputed['Age']).reshape(-1, 1), age_scale_param)

#print imputed.head()

cabin = pd.DataFrame()
# replacing missing cabins with U (for Uknown)
cabin['Cabin'] = full.Cabin.fillna('U')
# mapping each Cabin value with the cabin letter
cabin['Cabin'] = cabin['Cabin'].map( lambda c : c[0] )
# dummy encoding ...
cabin = pd.get_dummies(cabin['Cabin'], prefix='Cabin')
#print cabin.head()

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map( lambda t : t.strip(), ticket )
    ticket = list( filter( lambda t : not t.isdigit(), ticket ) )
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket['Ticket'] = full['Ticket'].map(cleanTicket)
ticket = pd.get_dummies(ticket['Ticket'], prefix='Ticket')

#print ticket.shape
#print ticket.head()

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

# introducing other features based on the family size
family['Family_Single'] = family['FamilySize'].map( lambda s :1 if s == 1 else 0 )
family['Family_Small'] = family['FamilySize'].map( lambda s :1 if 2 <= s <= 4 else 0 )
family['Family_Large'] = family['FamilySize'].map( lambda s :1 if s >= 5 else 0 )

#print family.head()
passengerId = pd.DataFrame()
passengerId['PassengerId'] = full['PassengerId']

# Select which features/variables to include in the dataset from the list below:
# imputed (Age, Fare), embarked , pclass , sex , family , cabin , ticket, title
full_X = pd.concat([passengerId, imputed, embarked, cabin, sex, pclass, family, title], axis=1)
#print full_X.head()

# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[0:891]
train_valid_Y = titanic.Survived

test_X = full_X[891:]
#print train_valid_X
trainFull_X, validFull_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_Y, train_size=0.7)

train_X = trainFull_X.ix[:,1:]
valid_X = validFull_X.ix[:,1:]
#print full_X.shape, train_X.shape, valid_X.shape, train_y.shape, valid_y.shape, test_X.shape

plot_variable_importance(train_X, train_y)

if True:
    # Random Forests Model 随机森林
    modelRFM = RandomForestClassifier(n_estimators=100)
    modelRFM.fit(train_X, train_y)
    print 'Random Forests Model 训练误差:', modelRFM.score(train_X, train_y), '测试误差:', modelRFM.score(valid_X, valid_y)

    #Support Vector Machines 支持向量机
    modelSVM = SVC()
    modelSVM.fit(train_X, train_y)
    print 'Support Vector Machines 训练误差:', modelSVM.score(train_X, train_y), '测试误差:', modelSVM.score(valid_X, valid_y)

    #Gradient Boosting Classifier
    modelGBC = GradientBoostingClassifier()
    modelGBC.fit(train_X, train_y)
    print 'Gradient Boosting Classifier 训练误差:', modelGBC.score(train_X, train_y), '测试误差:', modelGBC.score(valid_X, valid_y)

    #K-nearest neighbors
    modelKNN = KNeighborsClassifier(n_neighbors=3)
    modelKNN.fit(train_X, train_y)
    print 'K-nearest neighbors 训练误差:', modelKNN.score(train_X, train_y), '测试误差:', modelKNN.score(valid_X, valid_y)

    #Gaussian Naive Bayes
    modelGNB = GaussianNB()
    modelGNB.fit(train_X, train_y)
    print 'Gaussian Naive Bayes 训练误差:', modelGNB.score(train_X, train_y), '测试误差:', modelGNB.score(valid_X, valid_y)

    #Logistic Regression
    modelLR = LogisticRegression()
    modelLR.fit(train_X, train_y)
    print 'LR 训练误差:', modelLR.score(train_X, train_y), '测试误差:' , modelLR.score(valid_X, valid_y)
   # print pd.DataFrame({"columns": list(train_valid_X.columns)[1:], "coef": list(modelLR.coef_.T)})

    predictions = modelLR.predict(valid_X)
    bad_cases = titanic.loc[titanic['PassengerId'].isin(validFull_X[ predictions != valid_y]['PassengerId'].values)]
    bad_cases.to_csv(absPath + '/data/badcase.csv', index=False)
    print titanic.head()

    #test_Y = (testRFM_y + testSVM_y + testKNN_y + testGNB_y + testLR_y)
    #test_Y = np.where(test_Y > 2, 1, 0)
else:
    # Random Forests Model 随机森林
    modelRFM = RandomForestClassifier(n_estimators=100)
    modelRFM.fit(train_valid_X, train_valid_Y)
    print 'Random Forests Model 训练误差:', modelRFM.score(train_valid_X, train_valid_Y)
    testRFM_y = modelRFM.predict(test_X)

    # Support Vector Machines 支持向量机
    modelSVM = SVC()
    modelSVM.fit(train_valid_X, train_valid_Y)
    print 'Support Vector Machines 训练误差:', modelSVM.score(train_valid_X, train_valid_Y)
    testSVM_y = modelSVM.predict(test_X)

    # Gradient Boosting Classifier
    modelGBC = GradientBoostingClassifier()
    modelGBC.fit(train_valid_X, train_valid_Y)
    print 'Gradient Boosting Classifier 训练误差:', modelGBC.score(train_valid_X, train_valid_Y)
    testGBC_y = modelGBC.predict(test_X)

    # K-nearest neighbors
    modelKNN = KNeighborsClassifier(n_neighbors=3)
    modelKNN.fit(train_valid_X, train_valid_Y)
    print 'K-nearest neighbors 训练误差:', modelKNN.score(train_valid_X, train_valid_Y)
    testKNN_y = modelKNN.predict(test_X)

    # Gaussian Naive Bayes
    modelGNB = GaussianNB()
    modelGNB.fit(train_valid_X, train_valid_Y)
    print 'Gaussian Naive Bayes 训练误差:', modelGNB.score(train_valid_X, train_valid_Y)
    testGNB_y = modelGNB.predict(test_X)

    # Logistic Regression
    modelLR = LogisticRegression()
    modelLR.fit(train_valid_X, train_valid_Y)
    print 'LR 训练误差:', modelLR.score(train_valid_X, train_valid_Y)
    testLR_y = modelLR.predict(test_X)
    testLR_y = testLR_y.astype(np.int32)
    #print pd.DataFrame({"columns": list(train_valid_X.columns)[:], "coef": list(modelLR.coef_.T)})

    test_Y = (testSVM_y + testGNB_y + testLR_y)
    test_Y = np.where(test_Y > 1, 1, 0)

    passenger_id = full[891:].PassengerId
    test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': testLR_y } )
    test.shape
    test.head()
    test.to_csv( absPath +  '/data/titanic_pred.csv' , index = False )
