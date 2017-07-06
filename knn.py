from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
from operator import itemgetter

iris = load_iris()
# print iris['target_names']
fract = .8
print"training set",fract*100,'%     test set',(1-fract)*100,'%'
all_data = iris['data']
all_target = iris['target']

all_0 = all_data
all_0 = np.c_[all_0, all_target]

# print all_0
np.random.seed(0)
all_random = np.random.permutation(all_0)
# print all_random
size = int(fract * all_random[:, :1].size)
# print all_random[:, :1].size

train = all_random[:size, :]
test = all_random[size:, :]
# print test
train_size=train[:, :1].size
test_zize=test[:, :1].size



def distance1(datapoint1, datapoint2):
    dist = 0

    for i in range(len(datapoint1)):
        dist += math.pow((datapoint1[i] - datapoint2[i]), 2)
    return dist


def myneighbours(test1, k):
    dist = []
    for i in range(train_size):
        # print "klkl",test1[:3]
        d = distance1(train[i, :3], test1[:3])
        dist.append((train[i, :], d))
    # print "dist",dist[8]
    dist.sort(key=itemgetter(1))
    neighbours = []
    for j in range(k):
        neighbours.append(dist[j][0])
    return neighbours


def getRespose(neighbour):
    classVotes = {}
    for i in range(len(neighbour)):
        response = neighbour[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=itemgetter(1), reverse=True)
    return sortedVotes[0][0]


corr = 0
k=3
print"k value used",k
for x in range(test_zize):
    n = myneighbours(test[x], k)
    if getRespose(n) == test[x][-1]:
        corr += 1
        # print "predicted",getRespose(n),"expected",test[x][-1]

# print corr
print "eff from knn", float(corr) / test_zize * 100, "%"

knn = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
knn.fit(train[:, :3], train[:, 4])
pred = knn.predict(test[:, :3])
corr = 0
for i in range(test_zize):
    if (pred[i] == test[i][-1]):
        corr += 1
# print corr
print "eff of sklearn knn ", float(corr) / test_zize * 100, "%"

def logistic_regression():
    print ""