import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm

def rmse(t, y):
  return np.sqrt(np.mean((t - y)**2))

def simulate():
    df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
    df.head()
    df['LogPassengers'] = np.log(df['Passengers'])
    Ntest = 12
    train = df.iloc[:-Ntest]
    test = df.iloc[-Ntest:]
    model = pm.auto_arima(train['Passengers'],trace=True,suppress_warnings=True,seasonal=True, m=12)
    print(model.summary())

    test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['Passengers'], label='data')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index, confint[:,0], confint[:,1], color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig1.png")
    print("******************** fig1.png saved successfully ************************")

    train_pred = model.predict_in_sample(start=0, end=-1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Passengers'], label='data')
    ax.plot(train.index, train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig2.png")
    print("******************** fig2.png saved successfully ************************")

    logmodel = pm.auto_arima(train['LogPassengers'],trace=True,suppress_warnings=True,seasonal=True, m=12)
    print(logmodel.summary())

    test_pred_log, confint = logmodel.predict(n_periods=Ntest, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['LogPassengers'], label='data')
    ax.plot(test.index, test_pred_log, label='forecast')
    ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig3.png")
    print("******************** fig3.png saved successfully ************************")

    train_pred_log = logmodel.predict_in_sample(start=0, end=-1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['LogPassengers'], label='data')
    ax.plot(train.index, train_pred_log, label='fitted')
    ax.plot(test.index, test_pred_log, label='forecast')
    ax.fill_between(test.index, confint[:,0], confint[:,1], color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig4.png")
    print("******************** fig4.png saved successfully ************************")

    print("Non-logged RMSE:", rmse(test['Passengers'], test_pred))
    print("Logged RMSE:", rmse(test['Passengers'], np.exp(test_pred_log)))

    ### non-seasonal
    model = pm.auto_arima(train['LogPassengers'],trace=True,suppress_warnings=True,d=0,max_p=12, max_q=2, max_order=14,stepwise=False,seasonal=False)
    print(model.summary())

    test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['LogPassengers'], label='data')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig5.png")
    print("******************** fig5.png saved successfully ************************")

    train_pred = model.predict_in_sample(start=1, end=-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['LogPassengers'], label='data')
    ax.plot(train.index[1:], train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig6.png")
    print("******************** fig6.png saved successfully ************************")

    print("RMSE ERROR is: ",rmse(test['Passengers'], np.exp(test_pred)))

    ### non-seasonal non-logged
    model = pm.auto_arima(train['Passengers'],trace=True,suppress_warnings=True,max_p=12, max_q=12,max_order=24,stepwise=False,seasonal=False)
    print(model.summary())

    test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['Passengers'], label='data')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,confint[:,0], confint[:,1],color='red', alpha=0.3)
    ax.legend();
    plt.savefig("fig7.png")
    print("******************** fig7.png saved successfully ************************")

    print("RMSE ERROR is: ",rmse(test['Passengers'], test_pred))

if __name__ == '__main__':
    simulate()
