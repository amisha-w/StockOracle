import numpy as np
from datetime import datetime
import smtplib
import time
from selenium import webdriver
#For Prediction
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
#For Stock Data
from iexfinance import Stock
from iexfinance import get_historical_data

def getStocks(n):
    #Navigating to the Yahoo stock screener
    dir = os.path.dirname(__file__)
    chrome_driver_path = dir + "\chromedriver.exe"
    driver = webdriver.Chrome(chrome_driver_path)
    url = 'https://finance.yahoo.com/screener/predefined/aggressive_small_caps?offset=0&count=202'
    driver.get(url)

    stock_list = []
    n += 1
    for i in range(1, n):
    ticker = driver.find_element_by_xpath(
    '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(i) +          ']/td[1]/a')
    stock_list.append(ticker.text)

    #Using the stock list to predict the future price of the stock a specificed amount of days
    for i in stock_list:
    try:
        predictData(i, 5)
    except:
        print("Stock: " + i + " was not predicted")