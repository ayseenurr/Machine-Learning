# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:46:03 2020

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("satislar.csv")

X=data[['Aylar']]
y=data[['Satislar']]



#Verileri eğitim ve test kümesine ayırıyoruz

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


#
'''
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

y_train=sc.fit_transform(y_train)
y_test=sc.fit_transform(y_test)
'''

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

pred=lr.predict(X_test)

X_train=X_train.sort_index() # scaler işlemi yapılınca array oluyor o yüzden sort yapılamıyor 
y_train=y_train.sort_index()



plt.plot(X_train,y_train)
plt.plot(X_test,lr.predict(X_test))
plt.title('aylara göre satış')
plt.xlabel('aylar')
plt.ylabel('satışlar')

