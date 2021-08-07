import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

data=pd.read_csv('Salary_Data.csv')

X=pd.DataFrame(data.iloc[:,0])
y=data.iloc[:,1]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X,y)

#Loading our regressor model into a pickle file.... 
#We are actually writing our regressor into that pickle file 
pickle.dump(regressor,open('model.pkl','wb'))

#this pickle file will be deployed

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[1.1]]))