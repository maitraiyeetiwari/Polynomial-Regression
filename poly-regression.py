import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('fuelcons.csv')
df.columns

#check how engine size looks like when compared with co2emission
plt.scatter(df[['ENGINESIZE']],df[['CO2EMISSIONS']],color='blue')
plt.savefig('data.png',dpi=300)
#plt.show()

#we see that the relation is not really linear and can be better fit with a polynomial of degree 2

#make train and test data

msk=np.random.rand(len(df)) <0.8
train=df[msk]
test=df[~msk]

train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])

#but we want enginesize data to be in a way that there are degree 2 polynomial

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

#we make engine size into polynomial of degree 2
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

#now just use linear_model
regr=linear_model.LinearRegression()
regr.fit(train_x_poly,train_y)
regr.coef_
regr.intercept_

#evaluation metrics
from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
print('r2_score=',r2_score(test_y,regr.predict(test_x_poly)))


#see how the data looks like
plt.scatter(df[['ENGINESIZE']],df[['CO2EMISSIONS']],color='blue')
x=np.arange(0,10,0.1)
plt.plot(x, 104.93+(x*51.43)+((-1.6)*(x**2)),color='red',label='model=polynomial of degree 2')
plt.legend()
plt.xlabel('engine size')
plt.ylabel('CO2 emission')
plt.savefig('poly-regr.png', dpi=300)
#plt.show()






