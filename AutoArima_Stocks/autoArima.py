import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm

i = 1
Ntest = 30

def plot_result(model, fulldata, train, test):
  global Ntest
  params = model.get_params()
  d = params['order'][1] # list is in [p,d,q] form

  train_pred = model.predict_in_sample(start=d, end=-1)
  test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)

  print("first data:", fulldata.index[0])
  print("first train:", train.index[d])

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(fulldata.index, fulldata, label='data')
  ax.plot(train.index[d:], train_pred, label='fitted')
  ax.plot(test.index, test_pred, label='forecast')
  ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
  ax.legend();
  plt.savefig("fig{}.png".format(i))
  print("******************** fig{}.png saved successfully ************************".format(i))


def plot_test(model, test):
  global Ntest
  test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(test.index, test, label='true')
  ax.plot(test.index, test_pred, label='forecast')
  ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
  ax.legend();
  plt.savefig("fig{}.png".format(i))
  print("******************** fig{}.png saved successfully ************************".format(i))

def rmse(y, t):
  return np.sqrt(np.mean((t - y)**2))

def simulate():
    global i, Ntest
    df = pd.read_csv('sp500sub.csv', index_col='Date', parse_dates=True)
    df.head()
    goog = df[df['Name'] == 'GOOG']['Close']
    # goog = np.log(df[df['Name'] == 'GOOG']['Close'])
    goog.plot();
    train = goog.iloc[:-Ntest]
    test = goog.iloc[-Ntest:]
    model = pm.auto_arima(train,error_action='ignore', trace=True,suppress_warnings=True, maxiter=10,seasonal=False)
    print(model.summary())
    model.get_params()
    plot_result(model, goog, train, test)
    i+=1
    plot_test(model, test)
    i+=1
    print("RMSE ARIMA:", rmse(model.predict(Ntest).to_numpy(), test.to_numpy()))
    print("RMSE Naive:", rmse(train.iloc[-1], test)) # last known train data is naive forecast
    aapl = df[df['Name'] == 'AAPL']['Close']
    # aapl = np.log(df[df['Name'] == 'AAPL']['Close'])
    aapl.plot();
    train = aapl.iloc[:-Ntest]
    test = aapl.iloc[-Ntest:]
    model = pm.auto_arima(train,error_action='ignore', trace=True,suppress_warnings=True, maxiter=10,seasonal=False)
    print(model.summary())
    plot_result(model, aapl, train, test)
    i+=1
    plot_test(model, test)
    i+=1
    print("RMSE ARIMA:", rmse(model.predict(Ntest).to_numpy(), test.to_numpy()))
    print("RMSE Naive:", rmse(train.iloc[-1], test))
    ibm = df[df['Name'] == 'IBM']['Close']
    # ibm = np.log(df[df['Name'] == 'IBM']['Close'])
    ibm.plot();
    train = ibm.iloc[:-Ntest]
    test = ibm.iloc[-Ntest:]
    model = pm.auto_arima(train,error_action='ignore', trace=True,suppress_warnings=True, maxiter=10,seasonal=False)
    print(model.summary())
    plot_result(model, ibm, train, test)
    i+=1
    plot_test(model, test)
    i+=1
    print("RMSE ARIMA:", rmse(model.predict(Ntest).to_numpy(), test.to_numpy()))
    print("RMSE Naive:", rmse(train.iloc[-1], test))
    sbux = df[df['Name'] == 'SBUX']['Close']
    # sbux = np.log(df[df['Name'] == 'SBUX']['Close'])
    sbux.plot();
    train = sbux.iloc[:-Ntest]
    test = sbux.iloc[-Ntest:]
    model = pm.auto_arima(train,error_action='ignore', trace=True,suppress_warnings=True, maxiter=10,seasonal=False)
    print(model.summary())
    plot_result(model, sbux, train, test)
    i+=1
    plot_test(model, test)
    i+=1
    print("RMSE ARIMA:", rmse(model.predict(Ntest).to_numpy(), test.to_numpy()))
    print("RMSE Naive:", rmse(train.iloc[-1], test))

if __name__ == '__main__':
    simulate()
