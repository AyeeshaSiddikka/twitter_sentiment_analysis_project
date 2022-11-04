"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import datetime
import tweepy
from polygon import RESTClient
import pickle
import re

import emoji

img, header = st.columns([2, 10])
with img:
    st.image('./bull_vs_bear.jpg')
with header:
    st.header('Twitter Sentiment Analysis for Stocks')

stock_names = ['AMZN', 'AMT', 'AAPL', 'APRN', 'C', 'DIS', 'META', 'GE', 'GOOGL', 'HTZ', 'LUV', 'MSFT', 'NFLX', 'SBUX', 'TSLA']

def get_secrets(filename):
    secrets_file = open(filename, "r")
    string = secrets_file.read()
    string.split('\n')

    secrets_dict={}
    for line in string.split('\n'):
        if len(line) > 0:
            secrets_dict[line.split('=')[0]]=line.split('=')[1]
    return secrets_dict

secrets = get_secrets('secrets.txt')
client = tweepy.Client(secrets['twitter_bearer_token'])

# @st.cache
def get_tweets(stock_symbol, start_date_time, end_date_time):
    query = stock_symbol + " -is:retweet lang:en"
    response = []
    for tweet in tweepy.Paginator(client.search_recent_tweets,
                                 query = query,
                                 tweet_fields = ['created_at'],
                                 start_time = start_date_time,
                                 end_time = end_date_time,
                                 max_results=100).flatten(limit=1000):
        response.append(tweet)
    tweets = [tweet.text for tweet in response]
    date = [tweet.created_at.date() for tweet in response]
    tweets_df = pd.DataFrame({'text': tweets ,'date':date})
    return tweets_df

def get_max_min_datetime(selected_date):
    start_date_time = datetime.datetime.combine(selected_date, datetime.datetime.min.time())
    end_date_time = datetime.datetime.combine(selected_date, datetime.datetime.max.time())
    return (start_date_time, end_date_time)

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

rest_client = RESTClient(secrets["polygonio_api_key"])

@st.cache
def get_stock_open_close(stock_name, date):
    stock_data = {
        "date":[],
        "open":[],
        "high":[],
        "low":[],
        "close":[],
        "volume":[],
        "actual_sentiment": []
    }
    days = datetime.date.today() - date
    dates = [date + datetime.timedelta(days = i) for i in range(days.days)]
    dates = [d for d in dates if d.weekday() < 5][:3]
    for d in dates:
        response_poly = rest_client.get_daily_open_close_agg(stock_name, d.strftime('%Y-%m-%d'))
        stock_data["date"].append(datetime.datetime.fromisoformat(response_poly.from_).date())
        stock_data["open"].append(response_poly.open)
        stock_data["high"].append(response_poly.high)
        stock_data["low"].append(response_poly.low)
        stock_data["close"].append(response_poly.close)
        stock_data["volume"].append(response_poly.volume)
        stock_data["actual_sentiment"].append('bearish' if((response_poly.open - response_poly.close) > 0) else 'bullish')
    df = pd.DataFrame(stock_data)
    return df

preprocessor = pickle.load(open('cv_preprocessor.pkl', 'rb'))
model = pickle.load(open('cv_log_reg_model.pkl', 'rb'))

def get_tweet_sentiments(df):
    result = model.predict(preprocessor.transform(df['text']))
    df['sentiment'] = result
    return df

tab1, tab2 = st.tabs(["Tweet Prediction", "Analyze Tweets"])

with tab1:
    with st.form(key='input-form'):
        text = st.text_area(label = 'Enter some text related to any stock', value = "$AAPL The relief rally is reaching the resistance level, there's no divergence formed yet, so this is going down and retest the previous support level.")
        submitted_input_form = st.form_submit_button('Predict Sentiment')
    
    if submitted_input_form:
        df = pd.DataFrame({'text': [text]})
        sentiment = get_tweet_sentiments(df)['sentiment'].values[0]
        st.write('Sentiment of the above text is: ' + sentiment)

with tab2:
    with st.form(key='stock-form'):
        selected_stock_name = st.selectbox(label = 'Select a stock ticker', options = stock_names)
        selected_date = st.date_input('Select a date from last 7 days', value = yesterday, min_value = yesterday - datetime.timedelta(days=5), max_value = yesterday)
        submitted = st.form_submit_button('Predict')

    start_date_time, end_date_time = get_max_min_datetime(selected_date)
#     st.write('Start date is', start_date_time)
#     st.write('End date is', end_date_time)

    if submitted:
        tweets_df = get_tweet_sentiments(get_tweets(selected_stock_name, start_date_time, end_date_time))
        tweets_df

        predicted_sentiment = tweets_df['sentiment'].mode().values[0]
        st.header('Sentiment predicted: ' + predicted_sentiment)

        stock_df = get_stock_open_close(selected_stock_name, selected_date)
        if(selected_date.weekday() in [5, 6]):
            st.info('The selected day is in weekend, so showing prices from next working day!', icon="ℹ️")
        stock_df

        

        

        


