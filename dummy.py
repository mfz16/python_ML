# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
# y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
# z = np.polyfit(x, y, 3)
# p = np.poly1d(z)
# p30 = np.poly1d(np.polyfit(x, y, 30))
# xp = np.linspace(-2, 6, 100)
#
# print(np.linspace(1,10,19))
# _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
# plt.ylim(-2,2)
# #plt.xlim(-20,20)
# plt.show()

# import numpy as np
# import math
# from scipy import stats as st
# import sys
# n=int(sys.stdin.readline())
# x=np.array(sys.stdin.readline().split())
# x=x.astype(int)
# mean=np.mean(x)
# print mean
# mode=st.mode(x)
# median=np.median(x)
# print median
# print float(mode[0])
# sd=np.std(x)
# print sd
# ci=st.norm.interval(0.95,loc=mean,scale=sd/math.sqrt(n))
# print ("%f %f"%(ci[0],ci[1]))

# import sys
#
# n = int(sys.stdin.readline())
# ar = []
# for i in range(n):
#     ar.append(sys.stdin.readline().split())
# print ar[2][0]
# for i in range(len(ar)):
#     ar[i][0] = int(ar[i][0])
#     ar[i][1] = int(ar[i][1])
#     ar[i][2]= int(ar[i][2])
# print ar[1:3][2:3]

import math
# N = int(raw_input())
#
# grades = []
# for _ in range(N):
#     grades.append([int(b) for b in raw_input().split('\t')])
#
# m = [int(b[0]) for b in grades]
# p = [int(b[1]) for b in grades]
# c = [int(b[2]) for b in grades]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot

stock_price = pd.read_csv("E:/datasets/stock_prices.txt", delimiter=' ', index_col=None, header=None)

# print("st ",stock_price)
# print stock_price.ix[0:0,1:]
# print stock_price[1]
# print("orig", stock_price.index)
stock_price = stock_price.T
# print("transpose", stock_price.index)
stock_price.columns = stock_price.iloc[0]
stock_price = stock_price[1:]
print len(stock_price)
stock_price['Day'] = np.arange(1, (len(stock_price) + 1))
stock_price = stock_price.set_index('Day')

# print("rename ", stock_price.head)
# for i in stock_price.columns:
#     plt.plot(stock_price[i],label=i)
#     plt.plot(pd.rolling_mean(stock_price[i],30),label='moving average')
#     plt.legend(loc='best')
#     plt.show()
#     print i
# plt.show()
# print stock_price['UCSC']

int_money = 5000.00

stock_owned = 0

stocks_name=list(stock_price)
def Trade(by, curr, mon, stock_own):
    prev_mon=mon
    if (by == 1):

        while (mon > prev_mon/2):
            mon = mon - curr
            # print "st",stock_own
            stock_own = stock_own + 1
    elif (by == 0):
        print "bi is", by
        while (stock_own > 0):
            mon += curr
            stock_own = stock_own - 1

    return mon,prev_mon, stock_own


def Predic(stock_name,int_mon,st_valu):
    X = stock_price[stock_name]
    act='No'
    money=int_mon
    no_of_stocks=st_valu
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    model = 0
    model_fit = 0
    print(len(train))
    print (test.index)
    autocorrelation_plot(train)
    plt.show()
    m=[]
    m.append(int_money)
    buy, buyr = 0, 0
    correct_decisions = 0.0
    for t in range(len(test)):
        print "\nDay ", t
        model = ARIMA(history, order=(2, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        # print test.iloc[9]
        obs = test.iloc[t]

        history.append(obs)

        # print(history)
        print('current_price=%f,predicted=%f, expected=%f' % (history[-2], yhat, obs))

        if yhat > history[-2]:
            pred_act = 'Buy predicted'
            buy = 1

        else:
            pred_act = 'Sale predicted'
            buy = 0

        if obs > history[-2]:
            real_act = 'Buy expected'
            buyr = 1

        else:
            real_act = 'Sale expected'
            buyr = 0
        # error = mean_squared_error(test, predictions)
        if buy == buyr:
            correct_decisions += 1
        print('predicted=%s, expected=%s' % (pred_act, real_act))
        money,previous_money,no_of_stocks = Trade(buy, history[-2], money, no_of_stocks)
        m.append(money)


        if ((money >previous_money)):

            act = 'SELL'
        elif ((money <previous_money)):

            act = 'BUY'
        else:

            act = 'None'
        #print "previous_money ",previous_money
        print("Stock Name:",stock_name)
        print("Action  ", act)
        print("stocks owned:", no_of_stocks)
        print ("Current Cash:\n", money)


    from sklearn.metrics import mean_squared_error
    plt.plot(m)
    plt.show()
    mse = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % mse)
    # plot
    print ("correct decisions=%f percent" % (correct_decisions / len(test) * 100))
    plt.plot(X)
    plt.plot(test.index, predictions, color='red')
    plt.show()

Predic(stocks_name[0],int_money,stock_owned)