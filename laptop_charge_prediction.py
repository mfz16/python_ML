import numpy as np
data=np.genfromtxt("E:/datasets/charging_data.txt",delimiter=',')
import matplotlib.pyplot as plt

data_new=np.empty(0)
for i in range(100):
    if data[i][0]<4:
        data_new=np.append(data_new,data[i:i+1,0:2])
print data_new.shape
s=int(len(data_new)/2)
print s
data_new=np.reshape(data_new,(s,2))
print ("ghg",data_new)
data=data_new


size=int(.7*len(data))
x_train=data[:size,:1]
#x_train=x_train.reshape(len(x_train))

x_test=data[size:len(data),:1]
#x_test=x_test.reshape(len(x_test))

y_train=data[:size,1:2]
#y_train=y_train.reshape(len(y_train))

y_test=(data[size:len(data),1:2])
#y_test=y_test.reshape(len(y_test))

print x_train.shape
y_train.shape
plt.scatter(x_train,y_train,color='red',s=10)
plt.xlabel('charging')
plt.ylabel('life')
plt.show()
from sklearn import linear_model as lr
clf=lr.LinearRegression()
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, clf.predict(x_test))
print round(mse,3)
for i in range(len(x_test)):
     print("predicted=%f original=%f" % (pred[i],  y_test[i]))
