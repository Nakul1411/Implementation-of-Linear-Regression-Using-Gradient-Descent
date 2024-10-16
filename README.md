# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To train, initialize theta and iteratively update it using gradient descent.
2. For preprocessing, read and scale data.
3. For modeling , train the linear regression model.
4. To predict data values ,scale a new data and perdict.
5. Print the prediction.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: NAKUL R
RegisterNumber: 212223240102
*/
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
![Screenshot 2024-10-16 110542](https://github.com/user-attachments/assets/2fe1df78-739d-4bfc-944a-ddc05e714357)
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
![Screenshot 2024-10-16 110615](https://github.com/user-attachments/assets/3e4eed01-c833-4b49-af90-d830d323fbca)
x.shape
![Screenshot 2024-10-16 110642](https://github.com/user-attachments/assets/7badce1e-2652-4b8d-a799-ceda53855ac4)
y.shape
![Screenshot 2024-10-16 110707](https://github.com/user-attachments/assets/e2d277d5-a8ab-4b50-b604-926623018cf5)
m=0
c=0
L=0.001 # learning rate
epochs=5000 # No.of iterations to be performed
n=float(len(x))
error=[]
# Performing Gradient Descent
for i in range(epochs):
  y_pred = m*x + c
  D_m = (-2/n)sum(x(y-y_pred))
  D_c = (-2/n)*sum(y-y_pred)
  m = m-L*D_m
  c = c-L*D_c
  error.append(sum(y-y_pred)**2)
print(m,c)
type(error)
print(len(error))
plt.plot(range(0,epochs),error)
![Screenshot 2024-10-16 110740](https://github.com/user-attachments/assets/1f73cfd8-bfa2-422b-a3e1-d64565f8fc94)

## Output:
![linear regression using gradient descent](sam.png)
![Screenshot 2024-10-16 110759](https://github.com/user-attachments/assets/7daa48b6-6236-4ea2-b983-22e61684c6f2)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
