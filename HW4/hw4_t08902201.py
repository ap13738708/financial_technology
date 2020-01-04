import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from talib import MA, STOCH
from mpl_finance import candlestick_ohlc
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU
import time

data = pd.read_csv('SPY.csv')
df_plot = data[(data['Date'] >= '2018-01-01') & (data['Date'] <= '2018-12-31')].copy()

# ----------------------Graph plotting-----------------------------------
df_plot['Date'] = df_plot['Date'].astype('datetime64[ns]')
df_plot['Date'] = df_plot['Date'].map(mdates.date2num)

# Candlestick chart with 2 moving average line(10days and 30days)
charts_plot, (ax, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(40, 30),
                                           gridspec_kw={'height_ratios': [3, 0.5, 1], 'hspace': 0})
candlestick_ohlc(ax, df_plot.values, width=1, colorup='g', colordown='r')

ma_10_plot = MA(df_plot['Close'].values, timeperiod=10, matype=0)
ma_30_plot = MA(df_plot['Close'].values, timeperiod=30, matype=0)
ax.xaxis_date()

ax.plot(df_plot['Date'], ma_10_plot, label='MA-10')
ax.plot(df_plot['Date'], ma_30_plot, label='MA-30')
ax.legend()

# KD line chart
k_plot, d_plot = STOCH(df_plot['High'].values,
                       df_plot['Low'].values,
                       df_plot['Close'].values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                       slowd_matype=0)
ax2.plot(df_plot['Date'], k_plot, label='K')
ax2.plot(df_plot['Date'], d_plot, label='D')
ax2.legend()

# Volume bar chart
pos = df_plot['Open'] - df_plot['Close'] > 0
neg = df_plot['Open'] - df_plot['Close'] < 0

ax3.bar(df_plot['Date'].values[pos], df_plot['Volume'].values[pos], color='green', width=0.9, edgecolor='black')
ax3.bar(df_plot['Date'].values[neg], df_plot['Volume'].values[neg], color='red', width=0.9, edgecolor='black')

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=14))
charts_plot.autofmt_xdate()
charts_plot.savefig('charts_plot.png')
plt.show()

# -------------------------Data preprocessing------------------------------

# Add technical features
df_train = data[data['Date'] < '2017-01-01'].copy()

ma_10_train = MA(df_train['Close'].values, timeperiod=10, matype=0)
ma_30_train = MA(df_train['Close'].values, timeperiod=30, matype=0)

k_train, d_train = STOCH(df_train['High'].values,
                         df_train['Low'].values,
                         df_train['Close'].values, fastk_period=5,
                         slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

df_train['MA10'] = ma_10_train
df_train['MA30'] = ma_30_train
df_train['K'] = k_train
df_train['D'] = d_train

df_test = data[data['Date'] >= '2017-01-01'].copy()

ma_10_test = MA(df_test['Close'].values, timeperiod=10, matype=0)
ma_30_test = MA(df_test['Close'].values, timeperiod=30, matype=0)
k_test, d_test = STOCH(df_test['High'].values,
                       df_test['Low'].values, df_test['Close'].values,
                       fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df_test['MA10'] = ma_10_test
df_test['MA30'] = ma_30_test
df_test['K'] = k_test
df_test['D'] = d_test

# Drop unrelated data
df_train = df_train.dropna()
df_train.drop(['Date'], axis=1, inplace=True)

df_test = df_test.dropna(axis=0)

df_test['Date'] = df_test['Date'].astype('datetime64[ns]')
df_test['Date'] = df_test['Date'].map(mdates.date2num)

test_date = df_test['Date'].copy()

df_test.drop(['Date'], axis=1, inplace=True)


# Normalize
def normalize(data_frame):
    df = data_frame.copy()

    return (df - df.min()) / (df.max() - df.min())


df_train = normalize(df_train)
df_test = normalize(df_test)

# spilt data into train set and test set
data_train = df_train.values
data_test = df_test.values

x_train, y_train, x_test, y_test = [], [], [], []
for i in range(30, data_train.shape[0]):
    x_train.append(data_train[i - 30:i])
    y_train.append(data_train[i, 0])

for i in range(30, data_test.shape[0]):
    x_test.append(data_test[i - 30:i])
    y_test.append(data_test[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)


# ------------------------Create models----------------------------------
def result_plot(history, model, name):
    model_plot, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(25, 7), gridspec_kw={'width_ratios': [1, 2]})
    y_pred = model.predict(x_test)

    ax4[0].plot(history['loss'])
    ax4[0].plot(history['val_loss'])
    ax4[0].title.set_text('model loss')
    ax4[0].set_ylabel('loss')
    ax4[0].set_xlabel('Epoch')
    ax4[0].legend(['Train', 'Validation'], loc='upper left')

    ax4[1].plot(test_date[30:], y_test)
    ax4[1].plot(test_date[30:], y_pred)
    ax4[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax4[1].xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax4[1].title.set_text(name)
    ax4[1].legend(['real', 'predict'], loc='upper left')
    plt.setp(ax4[1].xaxis.get_majorticklabels(), rotation=30)
    model_plot.tight_layout()
    model_plot.savefig('{}_64units.png'.format(name))
    plt.show()


# RNN
regressor_RNN = Sequential()
regressor_RNN.add(SimpleRNN(units=64, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])))
regressor_RNN.add(Dense(units=1))
regressor_RNN.compile(optimizer='adam', loss='mean_squared_error')
start_time = time.time()
RNN_history = regressor_RNN.fit(x_train, y_train, epochs=128,
                                batch_size=64, validation_data=(x_test, y_test))
RNN_time = time.time() - start_time
result_plot(RNN_history.history, regressor_RNN, 'Prediction by RNN')

# LSTM
regressor_LSTM = Sequential()
regressor_LSTM.add(LSTM(units=64, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])))
regressor_LSTM.add(Dense(units=1))
regressor_LSTM.compile(optimizer='adam', loss='mean_squared_error')

start_time = time.time()
LSTM_history = regressor_LSTM.fit(x_train, y_train, epochs=128,
                                  batch_size=64, validation_data=(x_test, y_test))
LSTM_time = time.time() - start_time

result_plot(LSTM_history.history, regressor_LSTM, name='Prediction by LSTM')

# GRU
regressor_GRU = Sequential()
regressor_GRU.add(GRU(units=64, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])))
regressor_GRU.add(Dense(units=1))
regressor_GRU.compile(optimizer='adam', loss='mean_squared_error')

start_time = time.time()
GRU_history = regressor_GRU.fit(x_train, y_train, epochs=128,
                                batch_size=64, validation_data=(x_test, y_test))
GRU_time = time.time() - start_time

result_plot(GRU_history.history, regressor_GRU, name='Prediction by GRU')

print('Execution time of each model:')
print('RNN = {}'.format(RNN_time))
print('LSTM = {}'.format(LSTM_time))
print('GRU = {}'.format(GRU_time))
