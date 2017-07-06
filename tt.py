import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot

def Trade(by, curr, mon, stock_own):
    prev_mon=mon
    buy_c=0
    sell_c=0
    if (by == 1):

        while (mon > 0):
            if((mon-curr)>0):

                mon = mon - curr

                stock_own = stock_own + 1
                buy_c += 1
            else:
                break

    elif (by == 0):
        #print "bi is", by
        while (stock_own > 0):
            mon += curr
            stock_own = stock_own - 1
            sell_c += 1
    # print "stock change price",curr
    # print "previous mon",prev_mon
    # print "current money",mon
    print "sellc",sell_c
    print "buy",buy_c
    return mon, prev_mon, stock_own,buy_c,sell_c




def printTransactions(m, k, d, name, owned, prices):

    days=0

    money=m
    X = []
    no_of_tran=0
    act = 'No'
    stock_name=[]
    stock_name.extend(name)
    no_of_stocks=[]
    no_of_stocks.extend(owned)
    stock_price=[]
    stock_price.extend(prices)
    # print "pr",prices
    # print "str",stock_price
    size = []
    train = []
    test = []
    history = []
    predictions = []




    for i in range(len(stock_name)):
        X.append(stock_price[i])

        size.append(int(len(X[i]) * 0.66))
        train.append(X[i][0:size[i]])
        test.append(X[i][size[i]:len(X[i])])
        history.append([x for x in train[i]])
        predictions.append(list())
        temp = []
        correct_decisions = 0.0

    output = []
    yhat = []
    obs = []
    model = []
    model_fit = []
    # print "\nDay ", t
    act=[]
    previous_money = money
    for i in range(len(stock_name)):
        # print history[i]
        model.append(ARIMA(X[i], order=(1, 0, 0)))
        print "ar",X[i]
        model_fit.append(model[i].fit(disp=0))
        output.append(model_fit[i].forecast())
        yhat.append(output[i][0])
        predictions[i].append(yhat[i])

        # print test.iloc[9]
        obs.append(test[i])

        history[i].append(obs[i])
        if i == 2:
            temp.append(obs[1])
        # print(obs[i])
        #print('current_price=%f,predicted=%f, expected=%f' % (X[i][-1], yhat[i], obs[i][-2]))

        if (yhat[i] > X[i][-1]):
            pred_act = 'Buy predicted'
            buy = 1
            # print ("kj", (len(test[0]) - 1))
        else:
            pred_act = 'Sale predicted'
            buy = 0

        if obs[i] > X[i][-2]:
            real_act = 'Buy expected'
            buyr = 1

        else:
            real_act = 'Sale expected'
            buyr = 0
        # error = mean_squared_error(test, predictions)
        if buy == buyr:
            correct_decisions += 1
        print('predicted=%s, expected=%s\n' % (pred_act, real_act))

        if buy==0:
            money, previous_money, no_of_stocks[i],buy_c,sell_c = Trade(buy, X[i][-1], money, no_of_stocks[i])
        if buy==1:
            temp_money, previous_money, no_of_stocks[i],buy_c,sell_c = Trade(buy, X[i][-1], previous_money, no_of_stocks[i])
            money+=temp_money
        if (sell_c>0):


            act.append('SELL')
            no_of_tran+=1
        elif(buy_c>0):

            act.append('BUY')

            no_of_tran += 1
        else:

            act.append('Non')
            already_traded = False
        # print "previous_money ",previous_money
    print no_of_tran
    for i in range(len(stock_name)):
        # print "temp",temp_money
        #print"lk",money

        if(act[i]!='Non'):
            print ("%s %s %i" % (stock_name[i], act[i], no_of_stocks[i]))
        # print("Stock Name:", stock_name[i])
        # print("Action  ", act[i])
        # print("stocks owned:", no_of_stocks[i])
        # print ("Current Cash:", money)
    no_of_tran = 0
    print "mon", money
    days+=1





if __name__ == '__main__':
    m, k, d = [float(i) for i in raw_input().strip().split()]
    k = int(k)
    d = int(d)
    names = []
    owned = []
    prices = []
    for data in range(k):
        temp = raw_input().strip().split()
        names.append(temp[0])
        owned.append(int(temp[1]))
        prices.append([float(i) for i in temp[2:7]])

    printTransactions(m, k, d, names, owned, prices)
