import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
from datetime import datetime
#rcParams['figure.figsize'] = 15, 6
# data = pd.read_csv('E:/datasets/web.csv')
# print data.head()
# print '\n Data Types:'
# print data.dtypes
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#data = pd.read_csv('E:/datasets/web.csv', parse_dates=[0], index_col='Month',date_parser=dateparse)
# print data.head()
# data.index
# ts = data["#Passengers"]
# ts.head(10)

dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%b-%y')
data = pd.read_csv('E:/datasets/aapl1.csv', parse_dates=[0], index_col='Date',date_parser=dateparse)
# data['Date']=pd.to_datetime(data['Date'])
# data=data.set_index('Date')

ts_all=data["Average"]
size=int(.7*len(ts_all))
ts=ts_all[:size]
train=ts_all[size:len(ts_all)]
print ts.index


#1. Specific the index as a string constant:
#print ts['2011-01-01']

#2. Import the datetime library and use 'datetime' function:
from datetime import datetime
# ts[datetime(1949,1,1)]
# ts['1949']
plt.plot(ts)
plt.show()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=30)
    rolstd = pd.rolling_std(timeseries, window=30)

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

#ts_log=math.sqrt(ts)
plt.subplot(3,1,1)
plt.plot(ts_log)
plt.subplot(3,1,2)
plt.plot(np.sqrt(ts),color='red')
plt.subplot(3,1,3)
plt.plot(np.power(ts,1/3),color='green')

plt.title("log /sqrt")
plt.show()

moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
print ts_log_moving_avg_diff.head(30)
ts_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = pd.ewma(ts_log, halflife=30)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='purple')
plt.show()

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show()

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,freq=30)

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

# predictions_ARIMA_train_log=pdS
# plt.plot(predictions_ARIMA,color='red')
# plt.plot(train)
# plt.show()