# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import numpy as np
from scipy.fftpack import fft, fftfreq
import math
import matplotlib.pyplot as plt
fo=open("E:/datasets/web2.txt",'r')
n=int(fo.readline())
y=np.empty(0)
x=np.arange(1,n+1)
for i in range(n):
    temp=int(fo.readline())
    y=np.append(y,temp)
x_train=x[0:int(0.7*n)]
x_test=x[int(0.7*n):]
y_train=y[0:int(0.7*n)]
y_test=y[int(0.7*n):]


plt.scatter(x_train,y_train)
fp=np.polyfit(x_train,y_train,3)
poly=np.poly1d(fp)
plt.plot(x_train,poly(x_train))
plt.scatter(x_test,y_test)
plt.plot(x_test,poly(x_test))

plt.show()
four=fft(x_train)
freq=fftfreq(x_train.shape[-1])
plt.plot(freq,four)
plt.show()
#--for i in range(n+1,n+30):
    #print(poly(i))
error=pow((poly(x_test)-y_test),2)
e=math.sqrt(np.sum(error))/error.size
for i in range(3),:
    print(y_test[i]),
    print(poly(x_test[i]))
print(e)
fo.close()

ty