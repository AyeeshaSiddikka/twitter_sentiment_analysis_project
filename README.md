## Sentiment analysis of stock related tweets

In this NLP project, I have used sentiment analysis techniques to analyze the sentiments of stock related tweets by training a NLP model. Using the trained NLP model, tweets can be classified to either `bullish` or `bearish` sentiment. 
To compare the results of the model to actual data, stock price trend (open - close) is calculated from the actual stock price data retrieved from stock API.

The methods of this project include the followings:

- Get data
- Exploratory data analyses
- Data cleaning
- Data visualization
- Data wrangling
- Build model
- Validate model
- Build an app which uses the trained model to predict sentiment

### Dependencies
To successfully run this project, install the following dependencies using your dependency manager (pip or conda)

- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- nltk
- sklearn
- gensim

TODO: Create requirements.txt file for easier dependencies management

### Source dataset
- https://www.kaggle.com/datasets/sohelranaccselab/stock-market-tweets-data-sentiment-analysis

### Project structure
- 01-prepare-sentiment-labeled-tweets.ipynb: This file contains the logic to label the dataset, as the dataset obtained from Kaggle, doesn't have sentiment labels. Here I have used BERT transformer pretrained models from huggingface library to classify the sentiment of tweets. The output from this step is further used in the downstream notebooks as input dataset
- 02-eda-sentiment-price.ipynb: In this notebook, I have performed data wrangling and Exploratory Data Analysis (EDA) to better understand the data
- 03-nlp-sentiment-model.ipynb - In this notebook, I have tried using different vectorization techniques to convert the words in tweets to vectors like
    1. CountVectorizer
    2. TfidfVectorizer
    3. Word2Vec
Before vectorizing texts, to reduce the number of features, I have done some pre-processing / cleaning of tweets to replace urls, @mentions, #tags with pre-defined tokens and removed punctuations and special characters. And then vectorization is done using the above mentioned techniques. Once vectorization is done, I have used the obtained vectors for every tweet as my `X` and used the sentiments (bullish or bearish) os my `y` to train a LogisticRegression model. And also validated the trained model with test dataset (splitted from original dataset)
- app.py: This file contains a frontend app built using streamlit. This app uses the trained model and pre-processor / vectorizer to predict the sentiment of a given text or to predict the sentiments of tweets for a selected stock on a given date

### Conclusion
After training my model with different vectorization techniques, I have compared the f1 score and accuracy of each techniqe, and I have observed that, there is not much difference between scores of CountVectorization and TfidfVectorizer. However Word2Vec technique has given comparitively lesser scores. It could be because the vectors of individual words are averaged to get final vector for a tweet, the sentiment is lost during this average. Further investigation might be needed to understand the Word2Vec results better.

### Future work
- The dataset can be preprocessed further to reduce the number of features and the model can be trained with different text cleaning approaches, like with punctuations and without punctuations, so that, for example, the model can find similarities between the word doesn't and doesnt
- Transformer based models can be used for sentiment analysis
