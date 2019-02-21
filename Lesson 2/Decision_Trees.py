# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 00:41:11 2019

@author: useraccountname
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df=np.asarray(pd.read_csv('data.csv',header=None))
x=df[:,0:2]
y=df[:,2]
#print(y)
model=DecisionTreeClassifier(max_depth=7,min_samples_leaf=1)

model.fit(x,y)
prediction=model.predict(x)

acc=accuracy_score(y,prediction)
print(acc)