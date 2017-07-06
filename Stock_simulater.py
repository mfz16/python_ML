import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot

stock_price = pd.read_csv("E:/datasets/stock_prices.txt", delimiter=' ', index_col=None, header=None)
int_money = 5000.00

#m.append(int_money/1000)
stock_owned = 0
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

stocks_name=list(stock_price)
def Trade(by, curr, mon, stock_own):
    prev_mon=mon
    if (by == 1):

        while (mon > 1000):
            mon = mon - curr
            # print "st",stock_own
            stock_own = stock_own + 1
    elif (by == 0):
        #print "bi is", by
        while (stock_own > 0):
            mon += curr
            stock_own = stock_own - 1

    return mon,prev_mon, stock_own


def Predic(stock_name,int_mon,st_valu):
    #X = stock_price[stock_name]
    X=[]
    no_of_tran=0
    print stock_name
    print len(stock_name)
    act='No'
    money=int_mon
    no_of_stocks=[0,0,0,0,0,0,0,0,0,0]
    size=[]
    train=[]
    test=[]
    history=[]
    predictions=[]
    out_data_col=['Day', 'Money','Transactions done']
    out_data_val=[0,5000,0]
    ou_tp=[]
    for i in range(len(stock_name)):
        X.append(stock_price[stock_name[i]])
        print "XX",X
        size.append( int(len(X[i]) * 0.66))
        train.append(X[i][0:size[i]])
        test.append(X[i][size[i]:len(X[i])])
        history.append( [x for x in train[i]])
        predictions.append(list())
        out_data_col.append(stock_name[i])
        out_data_val.append(0)

    print out_data_val
    print out_data_col
    out_data = pd.DataFrame([out_data_val], columns=out_data_col, index=None)
    print out_data

    model =[]
    model_fit =[]

    #previous_money=[]
    already_traded=False
    # print(len(train))
    # print (test.index)
    # autocorrelation_plot(train)
    # plt.show()
    buy, buyr = 0, 0
    temp=[]
    correct_decisions = 0.0
    for t in range(len(test[0])):
        output = []
        yhat = []
        obs = []
        model = []
        model_fit = []
        print "\nDay ", t
        for i in range(len(stocks_name)):

            model.append(ARIMA(history[i], order=(2, 1, 0)))
            model_fit.append(model[i].fit(disp=0))
            output.append(model_fit[i].forecast())
            yhat.append(output[i][0])
            predictions[i].append(yhat[i])
            # print test.iloc[9]
            obs.append(test[i].iloc[t])

            history[i].append(obs[i])
            if i==2:
                temp.append(obs[1])
            #print(yhat[i])
            print('current_price=%f,predicted=%f, expected=%f' % (history[i][-2], yhat[i], obs[i]))

            if (yhat[i] > history[i][-2] and t!=(len(test[0])-1)) :
                pred_act = 'Buy predicted'
                buy = 1
                print ("kj",(len(test[0])-1))
            else:
                pred_act = 'Sale predicted'
                buy = 0

            if obs[i] > history[i][-2]:
                real_act = 'Buy expected'
                buyr = 1

            else:
                real_act = 'Sale expected'
                buyr = 0
            # error = mean_squared_error(test, predictions)
            if buy == buyr:
                correct_decisions += 1
            print('predicted=%s, expected=%s\n' % (pred_act, real_act))

            money, previous_money, no_of_stocks[i] = Trade(buy, history[i][-2], money, no_of_stocks[i])


            if ((money >previous_money) ):

                act = 'SELL'
                no_of_tran+=1
            elif ((money <previous_money) ):

                act = 'BUY'
                already_traded=True
                no_of_tran += 1
            else:

                act = 'None'
                already_traded=False
            #print "previous_money ",previous_money
            print ("%s %s %i"%(stock_name[i],act,no_of_stocks[i]))
            print("Stock Name:",stock_name[i])
            print("Action  ", act)
            print("stocks owned:", no_of_stocks[i])
            print ("Current Cash:", money)
            ou_tp.append(no_of_stocks[i])
        already_traded=False

        print "no_of tr",no_of_tran
        tt=[t+1,round(money,2),no_of_tran]
        tt.extend(ou_tp)
        print tt
        #out_data= out_data.append(pd.DataFrame([[t+1,money]],columns=['Day','Money'],index=None))
        out_data = out_data.append(pd.DataFrame([tt], columns=out_data_col, index=None))
        ou_tp=[]
        tt=[]
        no_of_tran = 0




    # print(m)
    # plt.plot(m)
    out_data.to_csv("E:/datasets/stockmoney.csv", sep=',')
    plt.show()
    from sklearn.metrics import mean_squared_error
    print "temp",temp
    print len(temp)
    mse = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % mse)
    # plot
    print ("correct decisions=%f percent" % (correct_decisions / len(test) * 100))
    plt.plot(X)
    plt.plot(test.index, predictions, color='red')
    plt.show()

Predic(stocks_name,int_money,stock_owned)