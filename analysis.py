import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data=pd.read_csv('/content/diabetes1.csv')

data.describe()

data.info()
#EDA

data.head()
sns.countplot(x='Pregnancies',data=data)

data.Pregnancies.value_counts()

sns.countplot(x='Pregnancies',hue='Outcome',data=data)

sns.pairplot(data,hue='Outcome')

sns.histplot(x='Glucose',hue='Outcome',data=data)

sns.relplot(x='Glucose',y='BloodPressure',hue='Outcome',data=data)

sns.relplot(x='Glucose',y='SkinThickness',hue='Outcome',data=data)

sns.histplot(x='BloodPressure',hue='Outcome',data=data)

sns.relplot(x='BloodPressure',y='SkinThickness',hue='Outcome',data=data)
#no rel

sns.relplot(x='BloodPressure',y='Insulin',hue='Outcome',data=data)

sns.histplot(x='Insulin',hue='Outcome',data=data)

#Data Preprocessing and Feature Engineering.
data.isnull().sum()

data.Glucose.replace(0,np.median(data.Glucose),inplace=True)

#dataframe.colum.replace('Value to be replaced','By what value')
#handling the corrupted data

data.loc[data['BMI']==0]

#Checking the outliers
plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1

for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.boxplot(data[column])
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Count',fontsize=20)
    plotnumber+=1
plt.tight_layout()

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
d1=['Pregnancies','Outcome']
data1=scaler.fit_transform(data.drop(d1,axis=1))

con_data=data[['Pregnancies','Outcome']]

data.columns

type(data1)
data2=pd.DataFrame(data1,columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])

df=pd.concat([con_data,data2],axis=1)

df

#feature selection
sns.heatmap(df.corr(),annot=True)

#model creation
X=df.iloc[:,:-1]
y=df.Outcome

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

y_test

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

y_pred

y_pred_prob=lr.predict_proba(X_test)

y_pred_prob

data.Outcome.value_counts()

#testing errors according to the confusion matrix

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,recall_score,precision_score,f1_score

print(confusion_matrix(y_test,y_pred))

r=recall_score(y_test,y_pred)
r

p=precision_score(y_test,y_pred)
p

f = f1_score(y_test,y_pred)
f

cr=classification_report(y_test,y_pred)
print(cr)

y_test.value_counts()


