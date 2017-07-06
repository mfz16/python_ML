import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
data=np.array([(0.18, 0.89, 109.85),
               (1.0, 0.26, 155.72),
               (0.92, 0.11, 137.66),
               (0.07, 0.37, 76.17),
               (0.85, 0.16, 139.75),
               (0.99, 0.41, 162.6),
               (0.87, 0.47, 151.77)])



datapoints=np.array(data[:,:2])
poly=PolynomialFeatures(degree=6)
X=poly.fit_transform(datapoints)
#print datapoints
clf1 = linear_model.LinearRegression()
#print(data[:,2:3])
clf1.fit(X,data[:,2:3])
#print (clf1.coef_)

predict1=np.array([0.99, 0.41])
predict_=poly.fit_transform(predict1)
calc=clf1.predict(predict_)
print "ty"
n=round(calc,3)
print (n)