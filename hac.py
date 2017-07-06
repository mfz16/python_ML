import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


f=int(raw_input())
t=int(raw_input())
data = np.empty((0))
for i in range((f+1)*t):
    data = np.append(data, int(raw_input()))
datapoints=np.array(data[:,:2])
poly=PolynomialFeatures(degree=2)
X=poly.fit_transform(datapoints)
#print datapoints
clf1 = linear_model.LinearRegression()
#print(data[:,2:3])
clf1.fit(X,data[:,2:3])
#print
'3' \
''

predict1=np.array([0.99, 0.41])
predict_=poly.fit_transform(predict1)
calc=clf1.predict(predict_)
print "ty"
print (calc)