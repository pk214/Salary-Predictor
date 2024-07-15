import pandas as pd
df=pd.read_csv("Salary_Data.csv")
x=df.iloc[:,0:1].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                        test_size=.3,random_state=1)

from sklearn.linear_model import LinearRegression
rg=LinearRegression()
rg.fit(x_train,y_train)

y_pred=rg.predict(x_test)

s=float(input("Enter Your years of experience: "))

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
print("Mean Square Error= ",mse)

import numpy as np
rmse=np.sqrt(mse)
print("Root Mean Square Error= ",rmse)

m=rg.coef_
print("Coffecient= ",m)

c=rg.intercept_
print("Intercept= ",c)

sal=m*s+c
print("Salary= ",sal)

s1=rg.predict([[s]])
print("Now Salry=",s1)

import matplotlib.pyplot as plt

plt.plot(x_train,y_train,'r*')
pr_y=rg.predict(x_train)
plt.plot(x_train,pr_y)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Exp-salary")
plt.show()