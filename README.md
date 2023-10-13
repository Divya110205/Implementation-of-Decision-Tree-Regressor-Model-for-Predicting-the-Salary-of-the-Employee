# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets

2.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters

3.Train your model -Fit model to training data -Calculate mean salary value for each subset

4.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance

5.Tune hyperparameters -Experiment with different hyperparameters to improve performance

6.Deploy your model Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DIVYA.A
RegisterNumber: 2122222230034 
*/

import pandas as pd
df=pd.read_csv('/content/Salary.csv')

df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### data.head():
![1](https://github.com/Divya110205/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404855/277f4a3d-ca5f-45ce-a05c-662120e7f8bf)

### data.info():
![2](https://github.com/Divya110205/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404855/2282f991-cff7-402e-b81e-537414f47187)

### isnull() & sum() Function:
![3](https://github.com/Divya110205/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404855/daa60f87-01e0-4c9a-89a6-b385866cf05d)

### data.head() For Position:
![4](https://github.com/Divya110205/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404855/e8f4d6bf-edc8-4617-aeba-d1cfb4956e57)

### MSE Value:
![5](https://github.com/Divya110205/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404855/38860a8d-933f-40ae-9370-1ce11d25531a)

### R2 Value:
![6](https://github.com/Divya110205/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404855/c830cef2-24f3-457a-b098-036e53e486e3)

### Prediction Value:
![7](https://github.com/Divya110205/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404855/72078f50-19a0-47bd-8db0-80efd86d70f0)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
