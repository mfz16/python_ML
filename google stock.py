# import time
# import urllib2
# import json
# def fetchPreMarket(symbol, exchange):
#     link = "http://finance.google.com/finance/info?client=ig&q="
#     url = link + "%s:%s" % (exchange, symbol)
#     u = urllib2.urlopen(url)
#     content = u.read()
#     data = json.loads(content[3:])
#     info = data[0]
#     t = str(info["elt"])  # time stamp
#     l = float(info["l"])  # close price (previous trading day)
#     p = float(info["el"])  # stock price in pre-market (after-hours)
#     return (t, l, p)
#
#
# p0 = 0
# while True:
#     t, l, p = fetchPreMarket("AAPL", "NASDAQ")
#     if (p != p0):
#         p0 = p
#         print("%s\t%.2f\t%.2f\t%+.2f\t%+.2f%%" % (t, l, p, p - l,
#                                                   (p / l - 1) * 100.))
#     time.sleep(60)













import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot

stock_price = pd.read_csv("E:/datasets/INFY.csv", delimiter=',')
stock_price["Avg"]=(stock_price["High"]+stock_price["Low"])/2
print stock_price
print stock_price['Date']
stock_price['Date']=pd.to_datetime(stock_price['Date'])

print stock_price['Date']
#plt.plot(stock_price['Date'],stock_price['Fluct'])
#plt.show()
# stock_price.set_index('Date')
# print stock_price
int_money = 500000.00

stock_owned = 0

stocks_name=list(stock_price)
def Trade(by, curr, mon, stock_own):
    prev_mon=mon
    if (by == 1):

        while (mon > 0):
            if(mon-curr>0):
                mon = mon - curr
            # print "st",stock_own
                stock_own = stock_own + 1
            else:
                break
    elif (by == 0):
        print "bi is", by
        while (stock_own > 0):
            mon += curr
            stock_own = stock_own - 1

    return mon,prev_mon, stock_own


def Predic(stock_name,int_mon,st_valu):

    X = stock_price[stock_name]
    act='No'
    plt.plot(stock_price['Date'],X)
    plt.show()
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
        model = ARIMA(history, order=(2, 2, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        # print test.iloc[9]
        obs = test.iloc[t]

        history.append(obs)

        # print(history)
        print('current_price=%f,predicted=%f, expected=%f' % (history[-2], yhat, obs))

        if yhat > history[-2] and t<len(test)-1:
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
    plt.plot(train,color='blue')
    plt.legend()
    plt.plot(test.index,test,color='green')
    plt.plot(test.index, predictions, color='red')
    plt.show()

Predic("Avg",int_money,stock_owned)