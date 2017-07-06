from matplotlib import pyplot as plt
import numpy as np
#import pandas.rpy.common as rcom
import pandas as pd
#We load the data with load_iris from sklearn
from sklearn.datasets import load_iris
data = load_iris()


x=data.data[:,:2]  # we only take the first two features
#print (x[:,:2])

plt.scatter(x[:,0], x[:,1], s=10,color='red')
plt.title("graph1")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()
# load_iris returns an object with several fields
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names
# print target_names[target]

f,subplt=plt.subplots(2,3)
for j in range(2):
    for i in range(1,4):
        for t in range(3):
            if t == 0:
                c = 'r'
                marker = '>'

            elif t == 1:
                c = 'g'
                marker = 'o'
            elif t == 2:
                c = 'b'
                marker = 'x'
            if(j!=1 or (i-1)!=2):

                subplt[0+j,i-1].scatter(features[target == t,0+j],
                    features[target == t,i+j],
                    marker=marker,
                    c=c)
                subplt[0+j,i-1].set_xlabel(feature_names[0+j])
                subplt[0+j,i-1].set_ylabel(feature_names[i+j])
            else:

                subplt[0 + j, i - 1].scatter(features[target == t, 3],
                                                 features[target == t, 2],
                                                 marker=marker,
                                                 c=c)
                subplt[0 + j, i - 1].set_xlabel(feature_names[3])
                subplt[0 + j, i - 1].set_ylabel(feature_names[2])







# We use NumPy fancy indexing to get an array of strings:
labels = target_names[target]
# The petal length is the feature at position 2
plength = features[:, 2]
# Build an array of booleans:
is_setosa = (labels == 'setosa')
# This is the important step:
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))
# plt.xlabel(feature_names[0])
# plt.ylabel(feature_names[1])

setosa_max_sepal_width = features[target == 0, 0].max()
setosa_min_sepal_width = features[target==0, 0].min()
setosa_max_sepal_length = features[target == 0, 1].max()
setosa_min_sepal_length = features[target==0, 1].min()
print setosa_max_sepal_width
print features[target == t, 0]
#plt.plot([setosa_min_sepal_width,setosa_max_sepal_width],[setosa_min_sepal_length,setosa_max_sepal_length])
plt.show()
features=features[~is_setosa]
print features
labels=labels[~is_setosa]
print labels
is_virginica=(labels=='virginica')
print is_virginica
best_acc=-1.0
u=0
df=pd.DataFrame({'Sepal_length':features[:,0],'Sepal_width':features[:,1],'Petal_length':features[:,2],'Petal_width':features[:,3],'Flower':labels})
print "fd",df.iloc[5:8]

c1_center=4.7
c1=np.empty(0)
c2=np.empty(0)
c2_center=5.7

train=df.iloc[:35]
train=train.append(df.iloc[50:85])

test=df.iloc[35:50]
test=test.append(df.iloc[85:100])
print test.index.size

featu=['Sepal_length','Sepal_width','Petal_length','Petal_width','po']
for i in range(train.index.size):
    if(train['Flower'].iloc[i]=='versicolor'):
        c1=np.append(c1,train[featu[0]].iloc[i])
        c1 = np.append(c1, train[featu[1]].iloc[i])
        c1 = np.append(c1, train[featu[2]].iloc[i])
        c1 = np.append(c1, train[featu[3]].iloc[i])
        c1_center=c1.mean()
    elif(train['Flower'].iloc[i]=='virginica'):
        c2=np.append(c2,train[featu[0]].iloc[i])
        c2 = np.append(c2, train[featu[1]].iloc[i])
        c2 = np.append(c2, train[featu[2]].iloc[i])
        c2 = np.append(c2, train[featu[3]].iloc[i])
co=0
print (features[2:3,2])
print test
tester=train
for i in range(tester.index.size):
    temp=(tester[featu[0]].iloc[i]+tester[featu[1]].iloc[i]+tester[featu[2]].iloc[i]+tester[featu[3]].iloc[i])/4
    print temp
    if(abs(temp-c1_center)<abs(temp-c2_center)):
        print"predicted versicolor"
        print"expected",tester['Flower'].iloc[i]
        if(tester['Flower'].iloc[i]=='versicolor'):
            co+=1
    else:
        print"predicted virginica"

        print"expected", tester['Flower'].iloc[i]
        if (tester['Flower'].iloc[i]=='virginica'):
            co += 1

print tester.index.size,co, float(co)/tester.index.size*100.0
0/0





for fi in range(features.shape[1]):
    thresh=features[:,fi]
    # print "threshold",thresh
    for t in thresh:
    # Get the vector for feature `fi`
            feature_i = features[:, fi]
            print "threshold", thresh
            print "features",feature_i
            print "count",u
            u+=1
            # apply threshold `t`

            pred = (feature_i > t)
            print "pred" ,pred
            acc = (pred == is_virginica).mean()
            rev_acc = (pred == ~is_virginica).mean()
            if rev_acc > acc:
                reverse = True
                acc = rev_acc
            else:
                reverse = False
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
                best_reverse = reverse
                print "best acc", best_acc
                print "best fi", best_fi
                print "best t", t
                print "best reverse", best_reverse
print "best acc",best_acc
print "best fi", best_fi
print "best t" ,t
print "best reverse",best_reverse

def is_virginica_test(fi, t, reverse, example):

    "Apply threshold model to a new example"
    test = example[fi] > t
    if reverse:
        test = not test
        return test
print is_virginica_test(best_fi,best_t,best_reverse,features)
print labels
