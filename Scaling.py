import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LinearRegression

col_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv',names=col_names)

array = data.values
X = array[:,0:8]
Y = array[:,8]
print(X.shape, Y.shape)

scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)
print(rescaled_X)
model = LinearRegression()
model.fit(X,Y)

predicted_Y = model.predict(X)
y = (predicted_Y > 0.5).astype(int)
print(y)


# binarizer = Binarizer(threshold=0.5).fit(X)
# binary_X = Binarizer.transform(X)
# print(binary_X[0:5,:])

# scaler = StandardScaler().fit(X)
# rescaled_X = scaler.transform(X)
# print(rescaled_X)



