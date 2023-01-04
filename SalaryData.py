# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 18:05:08 2023

@author: Rahul
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Naive Bayes\\SalaryData_Test.csv")
df
df.info()
df.shape

df.head()
df.dtypes

#Finding the special characters in the data frame 
df.isin(['?']).sum(axis=0)
print(df[0:5])

df.native.value_counts()
df.native.unique()

df.workclass.value_counts()
df.workclass.unique()

df.occupation.value_counts()
df.occupation.unique()

df.sex.value_counts()


#finding categorical and numerical
cat = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(cat)))
print('The categorical variables are :\n\n', cat)

#find numerical variables
num = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(num)))
print('The numerical variables are :', num)


# check if there are any missing values missing values in categorical variables

df[cat].isnull().sum()

df[num].isnull().sum()

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

df["age"].hist()
df["educationno"].hist()
df["capitalgain"].hist()
df["capitalloss"].hist()
df["hoursperweek"].hist()


t1 = pd.crosstab(index=df["education"],columns=df["workclass"])
t1.plot(kind='bar')

t2 = pd.crosstab(index=df["education"],columns=df["Salary"])
t2.plot(kind='bar')


t3 = pd.crosstab(index=df["sex"],columns=df["race"])
t3.plot(kind='bar')


t4 = pd.crosstab(index=df["maritalstatus"],columns=df["sex"])
t4.plot(kind='bar')

# Outliers dectection and treating outliers
df.boxplot("age",vert=False)
Q1=np.percentile(df["age"],25)
Q3=np.percentile(df["age"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["age"]<LW
df[df["age"]<LW]
df[df["age"]<LW].shape
df["age"]>UW
df[df["age"]>UW]
df[df["age"]>UW].shape
df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))

df.boxplot("educationno",vert=False)
Q1=np.percentile(df["educationno"],25)
Q3=np.percentile(df["educationno"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["educationno"]<LW
df[df["educationno"]<LW]
df[df["educationno"]<LW].shape
df["educationno"]>UW
df[df["educationno"]>UW]
df[df["educationno"]>UW].shape
df["educationno"]=np.where(df["educationno"]>UW,UW,np.where(df["educationno"]<LW,LW,df["educationno"]))

df.boxplot("hoursperweek",vert=False)
Q1=np.percentile(df["hoursperweek"],25)
Q3=np.percentile(df["hoursperweek"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["hoursperweek"]<LW
df[df["hoursperweek"]<LW]
df[df["hoursperweek"]<LW].shape
df["hoursperweek"]>UW
df[df["hoursperweek"]>UW]
df[df["hoursperweek"]>UW].shape
df["hoursperweek"]=np.where(df["hoursperweek"]>UW,UW,np.where(df["hoursperweek"]<LW,LW,df["hoursperweek"]))

df.boxplot("capitalgain",vert=False)
Q1=np.percentile(df["capitalgain"],25)
Q3=np.percentile(df["capitalgain"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["capitalgain"]<LW
df[df["capitalgain"]<LW]
df[df["capitalgain"]<LW].shape
df["capitalgain"]>UW
df[df["capitalgain"]>UW]
df[df["capitalgain"]>UW].shape
df["capitalgain"]=np.where(df["capitalgain"]>UW,UW,np.where(df["capitalgain"]<LW,LW,df["capitalgain"]))

df.boxplot("capitalloss",vert=False)
Q1=np.percentile(df["capitalloss"],25)
Q3=np.percentile(df["capitalloss"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["capitalloss"]<LW
df[df["capitalloss"]<LW]
df[df["capitalloss"]<LW].shape
df["capitalloss"]>UW
df[df["capitalloss"]>UW]
df[df["capitalloss"]>UW].shape
df["capitalloss"]=np.where(df["capitalloss"]>UW,UW,np.where(df["capitalloss"]<LW,LW,df["capitalloss"]))


# Spliting 
X=df.iloc[:,0:13]
X.columns
Y=df["Salary"]

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# display categorical variables
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical

# display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical

#Data Transformation
#pip install category_encoders
import category_encoders as ce


encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train.head()

from sklearn.preprocessing import MinMaxScaler #fixed import

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#model fitting
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, y_train)

y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)

from sklearn.metrics import accuracy_score
print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))



from sklearn.naive_bayes import BernoulliNB
bn = BernoulliNB()

bn.fit(X_train, y_train)

y_pred_train = bn.predict(X_train)
y_pred_test = bn.predict(X_test)

from sklearn.metrics import accuracy_score
print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))



from sklearn.naive_bayes import MultinomialNB
mb = MultinomialNB()

mb.fit(X_train, y_train)

y_pred_train = mb.predict(X_train)
y_pred_test = mb.predict(X_test)

from sklearn.metrics import accuracy_score
print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))



from sklearn.svm import SVC

svc_class = SVC(kernel='linear')

svc_class.fit(X_train,y_train)

y_predict_train = svc_class.predict(X_train)
y_predict_test = svc_class.predict(X_test)


print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

#prediction
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)

print("trainin score:",accuracy_score(y_train, y_pred_train).round(2))
print("test score:",accuracy_score(y_test, y_pred_test).round(2))













