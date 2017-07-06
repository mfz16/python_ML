y=7
x=(5.0*y-33)/4
print x
y=(20.0*x-107)/9
print y

y=7
x=(9.0*y+107)/20
print x
y=(4*x+33)/5.0
print y

import numpy as np
v=np.linspace(-15,15,100)
u=(5.0*v-33)/4
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# plt.plot(u,v)
# plt.show()

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y=np.array([10,10,20,20,20,30,50,60,70,70,70,80,100])
#plt.plot(x,y)
#plt.show()
#plt.hist(y,bins=13,normed=True)
#t=mlab.normpdf(13,np.mean(y),np.var(y))
#k=plt.plot(13,t,'r--',linewidth=2)
plt.show()
dat=np.linspace(-3,3,100)
#plt.plot(dat)

plt.hist(y,bins=10)
import seaborn as sns
plt.show()
sns.kdeplot(y)
plt.show()


import scipy.stats as st
