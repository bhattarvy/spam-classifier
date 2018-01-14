#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 03:57:51 2018

@author: arvy
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 06:10:48 2018

@author: arvy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
data=pd.read_csv('/home/arvy/Documents/ML/datasets/spambase.csv')


y=data['1']
X=data.drop(['1'],axis=1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.20, random_state=42)

from sklearn.svm import SVC
s=SVC(C=10,gamma=0.001)
s.fit(X_train,y_train)
print(s.score(X_test,y_test))   

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))