import keras
# import keras.backend as T
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
from datetime import datetime
import smtplib
import time
import os
from selenium import webdriver
#For Prediction
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
#For Stock Data
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
import pandas_datareader
import pandas as pd
from pandas_datareader import data







def predictData(stock,days):
    start = datetime(2016, 1, 1)
    end = datetime.now()
    #Outputting the Historical data into a .csv for later use
    #df = get_historical_data(stock, start,output_format='pandas')
    df = pandas_datareader.data.get_data_yahoo(stock, start, end)
    print(stock)
    print("before",df.head(1))    
    # csv_name = ('Exports/' + stock + '_Export.csv')    
    # df.to_csv(csv_name)
    
    #creating dataframe
    data=df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Close'])
    #new_data=df.drop(df.columns.difference(['Date','Close']), 1, inplace=True)
    
    #print(type(new_data))
    #print(new_data.columns.tolist())
    for i in range(0,len(df)):
        #new_data['Date'][i] = data1['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    #setting index
    # new_data.index = new_data.Date
    # new_data.drop('Date', axis=1, inplace=True)
    #creating train and test sets
    dataset = new_data.values
    k=int(0.75*len(new_data))
    train = dataset[0:k,:]
    valid = dataset[k:,:]

    #dataset to train variables
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(1,len(train)):
        x_train.append(scaled_data[i-1:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #Building model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    inputs = new_data[len(new_data) - len(valid) - 1:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)


    
    forecast_time = int(days)
    X_prediction = np.array(df.drop(df.columns.difference(['Close']),1))[-forecast_time:]
    # X = np.array(df['Close'])
  
    # X_prediction = X[-forecast_time:]

    # X_prediction = np.array(X_prediction)

    print('Shape: ',X_prediction.shape)
    X_prediction = np.reshape(X_prediction, (X_prediction.shape[0],X_prediction.shape[1],1))
    closing_price = model.predict(X_prediction)
    print(closing_price)
    closing_price = scaler.inverse_transform(closing_price)
    print(closing_price)

    

predictData('CLF',2)