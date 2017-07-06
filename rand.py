import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from  datetime import date as dt
from pandas.tools.plotting import autocorrelation_plot
from datetime import timedelta





fo=open("E:/datasets/web3.txt","r")
output=np.array([1808,1454,1393,1733,1944,1911,1804,1525,573,576,740,760,784,746,713
                                ,598,619,711,766,716,803,718,562,499,573,746,679,658,694,545])
n = int(fo.readline())
y = np.empty(0)
x = np.arange(1, n+1)
for i in range(n):

    temp = int(fo.readline())
    y = np.append(y, temp)
dataframe=pd.DataFrame(y)
print dataframe.head()

plt.show()

date=list()

init_date=dt(2012,9,30)
for i in range(n):
    init_date=init_date+timedelta(days=1)
    date.append(init_date)

dateframe=pd.DataFrame(date)
print (dateframe.dtypes)
dateframe=dateframe.astype('datetime64[ns]')

print (dateframe.dtypes)

dateframe[1]=dataframe[0]
dateframe=dateframe.set_index(dateframe[0])
print dateframe.head()
print dateframe.dtypes

df=pd.DataFrame(columns=["ind","hits"])
df["ind"]=dateframe[0]
df["hits"]=dateframe[1]
print df.index


ts = df["hits"]
print ts.head(10)

print ts.index
print ts['2013-5-6']
Y = dataframe.values
size = int(len(Y) * 0.66)
train, test = Y[0:size], Y[size:len(Y)]
#history = [x for x in train]
predictions = list()
model=0
model_fit=0

plt.plot(ts)
plt.show(block=False)


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=100)
    rolstd = pd.rolling_std(timeseries, window=100)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput

test_stationarity(ts)

ts_log = np.log(ts)
plt.plot(ts_log)
plt.show()

moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show(block=False)

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)

#test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='purple')
plt.show()

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show(block=False)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

model = ARIMA(ts_log, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
plt.show()


model = ARIMA(ts_log, order=(0, 1, 2))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
plt.show()

model = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()










































# plt.scatter(x,y,s=10)
# plt.xticks([100,200,300,400,500,600],['100','200','300','400','500','600'])
# plt.show()
#
#
# plt.scatter(x[:180],y[:180],s=2)
# plt.show()
# autocorrelation_plot(Y)
# plt.show()
#
# train1,test1=Y,[1808,1454,1393,1733,1944,1911,1804,1525,
# 573,576,740,760,784,746,713,598,619,711,766,716,803,718,
# 562,499,573,746,679,658,694,545]
# fo.close()
# expected=np.empty(0)
# fo=open("E:/datasets/web3_out.txt","r")
# for i in range(30):
#
#     temp1 = int(fo.readline())
#     expected = np.append(expected, temp1)
#
# model = ARIMA(Y,order=(2,0,1))
# result = model.fit(disp=-1)
#
#
# residual=pd.DataFrame(result.resid)
#
# residual.plot()
# plt.show()
# residual.plot(kind='kde')
# plt.show()
#
# plt.plot(Y, 'b-', label='data')
# #plt.show()
# #print (range(result.k_ar, len(X)))
# plt.plot( result.fittedvalues, 'r-')
# out1=result.predict(0,29)
# #plt.show()
# print(out1)
# plt.plot(np.arange(len(Y),len(Y)+30),expected,'y-')
# plt.plot(np.arange(len(Y),len(Y)+30),out1,'g-')
#
# plt.show()
# fo.close()
#
#
