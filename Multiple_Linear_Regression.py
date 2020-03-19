# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:47:04 2020

@author: HP
"""

import pandas as pd
import numpy as np


hava=pd.read_csv('odev_tenis.csv')

print(hava.corr())

hava['windy']=pd.get_dummies(hava['windy'])
hava['play']=pd.get_dummies(hava['play'])


outlook = hava.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
outlook[:,0] = le.fit_transform(outlook[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)



c=pd.DataFrame(data=outlook,index=range(14),columns=['sunny','overcast','rainy'])

hava.drop('outlook',axis=1,inplace=True)


veri=pd.concat([c,hava],axis=1)


x=veri.drop('humidity',axis=1)
y=veri[['humidity']].values


from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)
print(y_pred)



#parametre seçimi için statsmodeli kullanıyoruz.P value değerine göre parametreleri eliyoruz.
import statsmodels.formula.api as sm

X=np.append(arr=np.ones((14,1)).astype(int),values=x,axis=1)
X_l=veri.iloc[:,[0,1,2,3,4,5]].values
r=sm.OLS(endog=y,exog=X_l).fit()
print(r.summary())

veri.drop('rainy',axis=1,inplace=True)

X=np.append(arr=np.ones((14,1)).astype(int),values=x,axis=1)
X_l=veri.iloc[:,[0,1,2,3,4]].values
r=sm.OLS(endog=y,exog=X_l).fit()
print(r.summary())


x_train=x_train.drop('rainy',axis=1)
x_test=x_test.drop('rainy',axis=1)


regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

