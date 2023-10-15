# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data
2. Define your model
3. Define your cost function
4. Define your learning rate
5. Train your model
6. Evaluate your model
7. Tune hyperparameters
8. Deploy your model

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed By : SHALINI VENKATESAN
Register Number : 212222240096
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

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

#### Initial data set:

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/52f6a7d7-f853-4bf1-ae0c-1aca13597023)

#### Data info:

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/f5864740-26fd-4f7c-bb4e-16c1afd524da)

#### Optimization of null values:

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/e25fa0a1-fb9b-40d3-8d93-d64c22dcd592)

#### Assignment of x and y values:

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/cf116655-723d-425a-94a2-3e382da03b46)

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/fc91003e-c024-4aad-8541-381e42383976)

#### Converting string literals to numerical values using label encoder:

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/f6b3ebad-7251-4674-8758-a897e74117d3)

#### Accuracy:

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/7bf8f162-c90e-48f5-bb97-327201060be1)

#### Prediction:

![image](https://github.com/shalini-venkatesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118720291/3a30683e-662f-40a0-8d3c-9357c9c4cf09)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
