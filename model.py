# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 09:31:37 2022

@author: shrik
"""

import pandas as pd
import pandas as ps
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('students_placement.csv')
X = df.drop(columns='placed')
y = df['placed']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

model = SVC(kernel='rbf')
model.fit(X_train,y_train)
print(accuracy_score(y_test,model.predict(X_test)))
pickle.dump(model,open('model.pkl','wb'))