# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 02:09:22 2019

@author: useraccountname
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


df=pd.read_csv('bmi_and_life_expentency.csv')
BMI=df[['BMI']]
LE=df[['Life expectancy']]


bmi_model=LinearRegression()
bmi_model.fit(BMI,LE)

print(bmi_model.predict(21.07))

df.plot(x='BMI',y='Life expectancy',style='o')

