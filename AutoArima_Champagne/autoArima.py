import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

Ntest = 12
df = pd.read_csv('perrin-freres-monthly-champagne.csv',index_col='Month',skipfooter=2,parse_dates=True,engine='python')
train = test = None
count = 1

# from previous
def plot_fit_and_forecast(result, d=0, col='Sales'):
  global Ntest, train, test, count
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(df[col], label='data')
  # plot the curve fitted on train set
  train_pred = result.fittedvalues
  ax.plot(train.index[d:], train_pred[d:], color='green', label='fitted')
  # forecast the test set
  prediction_result = result.get_forecast(Ntest)
  conf_int = prediction_result.conf_int()
  lower, upper = conf_int[f'lower {col}'], conf_int[f'upper {col}']
  forecast = prediction_result.predicted_mean
  ax.plot(test.index, forecast, label='forecast')
  ax.fill_between(test.index,lower, upper,color='red', alpha=0.3)
  ax.legend()
  plt.savefig("fig{}.png".format(count))
  print("******************** fig{}.png saved successfully ************************".format(count))
  count+=1
  return forecast

def simulate():
    global Ntest, train, test, count
    print(df.head())
    df.columns = ['Sales']
    df['Sales'].plot();
    df['LogSales'] = np.log(df['Sales'])
    df['LogSales'].plot();
    df.index.freq = 'MS'
    train = df.iloc[:-Ntest]
    test = df.iloc[-Ntest:]
    # boolean series to index df rows
    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]
    model = pm.auto_arima(train['LogSales'],trace=True,suppress_warnings=True,seasonal=True, m=12)
    # Since the model is seasonal, we won't plot or predict the first 12 values (since pmdarima will set them to 0)
    train_pred = model.predict_in_sample(start=12, end=-1)
    test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['LogSales'], label='data')
    ax.plot(train.index[12:], train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig{}.png".format(count))
    print("******************** fig{}.png saved successfully ************************".format(count))
    count+=1

    # Compute R^2
    print("R2 score = ",r2_score(test['Sales'], np.exp(test_pred)))

    # Best non-seasonal model
    model = pm.auto_arima(train['LogSales'],trace=True,max_p=12, max_q=2, max_order=14,suppress_warnings=True,stepwise=False,seasonal=False)
    train_pred = model.predict_in_sample(start=1, end=-1)
    test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['LogSales'], label='data')
    ax.plot(train.index[1:], train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig{}.png".format(count))
    print("******************** fig{}.png saved successfully ************************".format(count))
    count+=1

    print("R2 score = ",r2_score(test['Sales'], np.exp(test_pred)))

    plot_acf(df['LogSales']);
    plt.savefig("fig{}.png".format(count))
    print("******************** fig{}.png saved successfully ************************".format(count))
    count+=1

    # You'll get a weird sqrt error with default method
    plot_pacf(df['LogSales'], method='ols');
    plt.savefig("fig{}.png".format(count))
    print("******************** fig{}.png saved successfully ************************".format(count))
    count+=1

    df['LogSales'].diff().plot();
    plt.savefig("fig{}.png".format(count))
    print("******************** fig{}.png saved successfully ************************".format(count))
    count+=1

    plot_acf(df['LogSales'].diff().dropna());
    plt.savefig("fig{}.png".format(count))
    print("******************** fig{}.png saved successfully ************************".format(count))
    count+=1

    plot_pacf(df['LogSales'].diff().dropna(), method='ols');
    plt.savefig("fig{}.png".format(count))
    print("******************** fig{}.png saved successfully ************************".format(count))
    count+=1

    print("DICKY FULLER TEST : LogSales is NOT STATIONARY")
    print(adfuller(df['LogSales']))

    print("DICKY FULLER TEST : LogSales Difference is STATIONARY")
    print(adfuller(df['LogSales'].diff().dropna()))

    arima = ARIMA(train['LogSales'], order=(12,1,2))
    arima_result = arima.fit()
    forecast = plot_fit_and_forecast(arima_result, d=1, col='LogSales')

    print("R2 score = ",r2_score(test['Sales'], np.exp(forecast)))

if __name__ == '__main__':
    simulate()
