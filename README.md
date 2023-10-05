# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SHALINI VENKATESAN
RegisterNumber: 212222240096 
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()
data.tail()
data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png)

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/f3d9e2f4-b0aa-4bca-b14d-f2dc8c1dcd29)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/6ed1faf3-2437-43a8-9e35-b2015827ba1e)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/286dbcbe-b826-4cff-9728-ffaedaeab91b)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/8775c891-2ff3-48ed-b84f-9198b32c484d)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/1cc1d071-31a7-469f-b622-acd994aed8b9)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/861b17da-9196-4f78-8bb0-00d9125bdd45)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/6f1092e7-8348-4731-8592-967edffd7fe9)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/3bd0c818-3039-4f65-b1cf-39661e9401d9)
![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/a957aec6-0ac6-4192-a25e-70627604096a)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
