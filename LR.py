import numpy as np 
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
from sklearn.svm import SVR



def getStocks(n): 
     predictData('CLF',2)
  


def predictData(stock,days):
    start = datetime(2016, 1, 1)
    end = datetime.now()
    #Outputting the Historical data into a .csv for later use    
    df = data.get_data_yahoo(stock, start, end)
    #print(df.head())
    # Get the Adjusted Close Price 
    df = df[['Adj Close']] 
    # Take a look at the new data 
    #print(df.head())
    # A variable for predicting 'n' days out into the future
    forecast_out = 5 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    #print the new data set
    #print(df.tail())
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))
    #Remove the last '30' rows
    X = X[:-forecast_out]
    #print(X)
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    #print(y)
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    # svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    # svr_rbf_confidence = svr_rbf.score(x_test, y_test)
    # print("svm confidence: ", svr_rbf_confidence)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    lr_confidence = lr.score(x_test, y_test)
    print("lr confidence: ", lr_confidence)
    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    #print(x_forecast)
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)
    #print("lr ans")
    print(lr_prediction)
    # Print support vector regressor model predictions for the next '30' days
    # svm_prediction = svr_rbf.predict(x_forecast)
    # print("svm ans")
    # print(svm_prediction)


 

getStocks(5)



