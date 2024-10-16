# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To train, initialize theta and iteratively update it using gradient descent.
2. For preprocessing, read and scale data.
3. For modeling, train the linear regression model.
4. To predict data values, scale a new data and predict.
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
![WhatsApp Image 2024-10-16 at 11 05 46_c72b662b](https://github.com/user-attachments/assets/976ca6a3-a492-4402-8049-f52464eafcc3)
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
![WhatsApp Image 2024-10-16 at 11 06 20_5ffa57a8](https://github.com/user-attachments/assets/aecd14a8-8bee-4b1a-8b1a-a54d9f12262f)
x.shape
![Screenshot 2024-10-16 110642](https://github.com/user-attachments/assets/733b8f2e-16b4-44b9-b1bc-2bbd5b6f6eb6)
y.shape
![Screenshot 2024-10-16 110707](https://github.com/user-attachments/assets/37f15f6a-10ef-48ee-a981-12de7c846bcd)
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
![Screenshot 2024-10-16 110740](https://github.com/user-attachments/assets/8c451e0c-0b2e-4f9c-aa19-5bcb1cb5015b)
![Screenshot 2024-10-16 110759](https://github.com/user-attachments/assets/152a55c2-842c-4308-9304-c68b1df43e3e)

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
