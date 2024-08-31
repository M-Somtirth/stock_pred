from flask import Flask, request
from flask import jsonify
import csv
import io
import time
import gzip
import pandas as pd
import os
import requests
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pickle


app = Flask(__name__)

def pre_preprocessor(ticker,company_name):    
    CSV_URL ="https://api.tiingo.com/tiingo/daily/"+ticker+"/prices?startDate=2015-01-02&token=419f55d2ac8913929c3054133171783b621cdc06"

    with requests.Session() as s:
        try:
            download = s.get(CSV_URL)
            decoded_content = download.content.decode('utf-8')
            df = pd.read_json(io.StringIO(decoded_content))
            df.to_csv(str(company_name) + '_data.csv', index=False)
        except ValueError as e:
            print(f"ValueError: {e} for"+ ticker)
        except Exception as e:
            print(f"An error occurred: {e} for" + ticker)
    return

def preprocessing(ticker):
    df=pd.read_csv('Tickers.csv',index_col=False,encoding='ISO-8859-1')

    row_numbers = df.index[df['TICKER'] == ticker].tolist()
    try:
        
        if row_numbers:
            row_number = row_numbers[0]
            company_name = df.loc[row_number, 'COMPANY NAME']
            df_pearl=pd.read_csv(str(company_name)+'_data.csv',index_col='date',parse_dates=True)

            # df_pearl['high'].plot(figsize=(12,6))
            df_pearl.index = pd.to_datetime(df_pearl.index)
            df_pearl = df_pearl.asfreq('D')
            df_pearl['high'].fillna(method='ffill', inplace=True)
            df_pearl['low'].fillna(method='ffill', inplace=True)
            df_pearl['open'].fillna(method='ffill', inplace=True)
            df_pearl['close'].fillna(method='ffill', inplace=True)
            res=seasonal_decompose(df_pearl['high'],period=365)
            # res.plot();
        else:
            print("golmaal hai bhai sab golmaal hai")
    except:

        if row_numbers:
            row_number = row_numbers[0]
            company_name = df.loc[row_number, 'COMPANY NAME']
            pre_preprocessor(ticker,company_name)
            df_pearl=pd.read_csv(str(company_name)+'_data.csv',index_col='date',parse_dates=True)

            # df_pearl['high'].plot(figsize=(12,6))
            df_pearl.index = pd.to_datetime(df_pearl.index)
            df_pearl = df_pearl.asfreq('D')
            df_pearl['high'].fillna(method='ffill', inplace=True)
            df_pearl['low'].fillna(method='ffill', inplace=True)
            df_pearl['open'].fillna(method='ffill', inplace=True)
            df_pearl['close'].fillna(method='ffill', inplace=True)
            res=seasonal_decompose(df_pearl['high'],period=365)
            # res.plot();
        else:
            print("golmaal hai bhai sab golmaal hai")
    return df_pearl
        
def value_predictor(best_model,target,df_pearl):
    train_test_split=len(df_pearl)-201
    train_df=df_pearl[:train_test_split]
    test_df=df_pearl[train_test_split:]
    model_open=SARIMAX(df_pearl[target], order=best_model.order,seasonal_order=best_model.seasonal_order)
    res=model_open.fit()
    start=len(train_df)
    end= len(train_df)+len(test_df)+14
    pred=res.predict(start,end).rename('Prediction_'+target)
    # plt.figure(figsize=(14, 8))
    # ax = test_df[target].plot(legend=True)
    # pred.plot(legend=True, ax=ax)
    # plt.show()
    # plt.clf()
    target_real=test_df[target]
    target_pred=pred
    return target_real,target_pred

def pred_using_ticker(ticker,df_pearl):
    try:
        with gzip.open(ticker+'_open.pkl.gz', 'rb') as f:
            best_model_open = pickle.load(f)   
            print("not found")
            print(ticker+'_open.pkl.gz')
            print(os.path.exists(ticker+'_open.pkl.gz'))
            print(best_model_open)
            print("Model loaded successfully.")
            print("not found")
    except (FileNotFoundError,OSError, IOError, EOFError) as e:
        print(e)
        best_model_open = auto_arima(df_pearl['open'], seasonal=True, m=7, trace=True)
        with gzip.open(ticker+'_open.pkl.gz', 'wb') as f:
            pickle.dump(best_model_open, f)
    try:
        with gzip.open(ticker+'_close.pkl.gz', 'rb') as f:
            best_model_close = pickle.load(f)   
            print(best_model_close)
            print("Model loaded successfully.")
    except:
        best_model_close = auto_arima(df_pearl['close'], seasonal=True, m=7, trace=True)
        with gzip.open(ticker+'_close.pkl.gz', 'wb') as f:
            pickle.dump(best_model_close, f)
            
    try:
        with gzip.open(ticker+'_high.pkl.gz', 'rb') as f:
            best_model_high = pickle.load(f)   
            print(best_model_high)
            print("Model loaded successfully.")
    except:
        best_model_high = auto_arima(df_pearl['high'], seasonal=True, m=7, trace=True)
        with gzip.open(ticker+'_high.pkl.gz', 'wb') as f:
            pickle.dump(best_model_high, f)
            
    try:
        with gzip.open(ticker+'_low.pkl.gz', 'rb') as f:
            best_model_low = pickle.load(f)   
            print(best_model_low)
            print("Model loaded successfully.")
    except (FileNotFoundError,OSError, IOError, EOFError) as e:
        print(e)
        best_model_low = auto_arima(df_pearl['low'], seasonal=True, m=7, trace=True)
        with gzip.open(ticker+'_low.pkl.gz', 'wb') as f:
            pickle.dump(best_model_low, f)
    
    open_real,open_pred=value_predictor(best_model_open,'open',df_pearl)
    close_real,close_pred=value_predictor(best_model_close,'close',df_pearl)
    high_real,high_pred=value_predictor(best_model_high,'high',df_pearl)
    low_real,low_pred=value_predictor(best_model_low,'low',df_pearl)
    return open_real,open_pred,close_real,close_pred,high_real,high_pred,low_real,low_pred

def main1(ticker='ZS'):
    df_pearl=preprocessing(ticker)
    open_real,open_pred,close_real,close_pred,high_real,high_pred,low_real,low_pred=pred_using_ticker(ticker,df_pearl)
    return open_real,open_pred,close_real,close_pred,high_real,high_pred,low_real,low_pred

@app.route('/')
def hello_world():
    # print(main1())
    ticker = request.args.get('ticker')  # Extract the 'ticker' query parameter
    if ticker is None:
        return jsonify({"error": "Ticker parameter is missing"}), 400
    open_real,open_pred,close_real,close_pred,high_real,high_pred,low_real,low_pred=main1(ticker)
    open_pred_1=open_pred[-7:]
    close_pred_1=close_pred[-7:]
    high_pred_1=high_pred[-7:]
    low_pred_1=low_pred[-7:]
    response = {
    "open_pred": open_pred_1.to_dict(),
    "close_pred": close_pred_1.to_dict(),
    "high_pred": high_pred_1.to_dict(),
    "low_pred": low_pred_1.to_dict()
    }
    response = {k: {str(key): value for key, value in v.items()} for k, v in response.items()}
    return jsonify(response)