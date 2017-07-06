# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
import sys
import math
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
fo = open("C://Users/Mohd/PycharmProjects/python_practise/office_test_in", "r")
#print "Name of the file: ", fo.name



sum = 0
#f, n = (sys.stdin.readline()).split()
f, n = (fo.readline()).split()

f = int(f)
n = int(n)
data = np.empty(0)
for i in range(n):
    #data = np.append(data, (sys.stdin.readline().split()))
    data = np.append(data, (fo.readline().split()))
#t = int(sys.stdin.readline())
t = int(fo.readline())

topredict = np.empty(0)
for i in range(t):
    #topredict = np.append(topredict, (sys.stdin.readline().split()))
    topredict = np.append(topredict, (fo.readline().split()))

data = np.reshape(data, (-1, f + 1))
data = data.astype(np.float)

topredict = np.reshape(topredict, (-1, f))
topredict = topredict.astype(np.float)

datapoints = np.array(data[:, :f])
poly = PolynomialFeatures(degree=4)
X = poly.fit_transform(datapoints)

# print datapoints
clf = linear_model.LinearRegression()
# print(data[:,2:3])d
clf.fit(X, data[:, f:f + 1])
# print (clf1.coef_)
plt.scatter(data[:,:1],data[:,f:f+1])
#plt.plot(X,data[:,f:f+1])
plt.show()
fo.close()
fo = open("C://Users/Mohd/PycharmProjects/python_practise/office_test_out", "r")
for i in range(t):
    # predict1=topredict()
    predict_ = poly.fit_transform(topredict[i:i+1,:])
    calc = clf.predict(predict_)
    n1 = round(calc, 3)
    print (n1)
    y=float(fo.readline())

    sum=sum+math.pow((n1-y),2)
print("sum error",math.sqrt(sum)/t)
fo.close()
