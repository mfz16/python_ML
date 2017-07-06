import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from datetime import date
d=np.array([1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1
,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1
,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1
,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1
,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1
,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1,2,3,2,3,4,3,2,1,0,-1,0,1])
d=d.astype(np.float)
n=np.arange(d.shape[-1])
four=np.fft.fft(d)
freq=np.fft.fftfreq(d.shape[-1])

f1=np.polyfit(n,d,9)
f=np.poly1d(f1)
plt.subplot(3,1,1)
plt.plot(n,d)
plt.subplot(3,1,2)
plt.plot(n,f(n))
plt.subplot(3,1,3)
plt.plot(freq,four)

y = np.zeros(len(four))
#Y[important frequencies] = X[important frequencies]

plt.show()




fo=open("E:/datasets/web2.txt","r")
output=np.array([1808,1454,1393,1733,1944,1911,1804,1525,573,576,740,760,784,746,713
                                ,598,619,711,766,716,803,718,562,499,573,746,679,658,694,545])
n = int(fo.readline())
x = np.empty(0)
#x = np.arange(1, n + 1)
for i in range(n):

    temp = int(fo.readline())
    x = np.append(x, temp)
dataframe=pd.DataFrame(x)
print dataframe.head()

plt.show()

X = dataframe.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
model=0
model_fit=0
autocorrelation_plot(train)
plt.show()
y# for t in range(len(test)):
#     model = ARIMA(history, order=(1,1,0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print(history)
#     print('predicted=%f, expected=%f' % (yhat, obs))
# #error = mean_squared_error(test, predictions)
# #print('Test MSE: %.3f' % error)
# # plot
# plt.plot(test)
# plt.plot(predictions, color='red')
# plt.show()

train1,test1=X,[1808,1454,1393,1733,1944,1911,1804,1525,
573,576,740,760,784,746,713,598,619,711,766,716,803,718,
562,499,573,746,679,658,694,545]

# history1 = [x for x in train1]
# print(history1)
# predictions = list()
#
# for t in range(len(test1)):
#     model = ARIMA(history1, order=(2,1,0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     print("output is",output)
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test1[t]
#     history1.append(obs)
#     #print(history1)
#     print('predicted=%f, expected=%f' % (yhat, obs))
# plt.plot(test1)
# plt.plot(predictions, color='red')
# plt.show()

# for i in test1:
#     model = ARIMA(history, order=(4, 1, 0))
#     model_fit=model.fit(disp=0)
#     out=model_fit.fittedvalues
#     print(out)
#     h=out[0]
#     history.append(h)
#     print('predicted=%f, expected=%f' % (h, i))


model = ARIMA(X,order=(5,0,0),freq=())
result = model.fit()

plt.plot(X, 'b-', label='data')
#plt.show()
#print (range(result.k_ar, len(X)))
plt.plot( result.fittedvalues, 'r-')
out1=result.predict(0,29)
#plt.show()
print(out1)
plt.plot(np.arange(501,531),test1,'y-')
plt.plot(np.arange(501,531),out1,'g-')

plt.show()



