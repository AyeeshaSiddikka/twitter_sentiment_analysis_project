import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet

class Preprocessor:
    def __init__(
        self, 
        stopwords=stopwords.words('english'), 
        vectorizer = CountVectorizer(max_features=30000, ngram_range=(1,3))
    ):
        self.stopwords = stopwords
        self.vectorizer = vectorizer
        self.vectorizer_fitted = False
        self.wnl = WordNetLemmatizer()
        self.usernam_pattern = re.compile(r'(\@)(\S+)')
        self.twitter_handle_pattern = re.compile(r'(\#)(\S+)')
    
    def clean_tweet(self, texts):
        print('Cleaning...')
        cleaned_texts = []
        for text in texts:
            # Replace @usernames with the text mention_
            text = re.sub(self.usernam_pattern, r'mention_\2', text)
            # Replace #hashtags with the text hashtag_
            text = re.sub(self.twitter_handle_pattern, r'hashtag_\2', text)
            # remove URLs
            text = re.sub(r'https?://\S+', "", text)
            text = re.sub(r'www.\S+', "", text)
            # remove $cashtags
            text = re.sub(r'(\$)([A-Za-z]+)', r'cashtag_\2', text)
            # deemojize
            text = emoji.demojize(text, delimiters=("", " "))
            # remove punctuations
            # text = re.sub(r'[^\w\s]', '', text)
            # remove numbers
            # text = re.sub(r'[0-9]', ' ', text)
            # remove extra spaces
            text = re.sub(r' +', ' ', text)
            text = text.strip()
            cleaned_texts.append(text)

        return cleaned_texts
    
    def tokenize(self, texts):
        print('Tokenizing...')
        return [word_tokenize(text) for text in texts]

    def lemmatize(self, texts):
        tokens_list = self.tokenize(texts)
        print('Negation handling...')
        return [' '.join(self.negation_handler(text)) for text in tokens_list]
    
    # https://github.com/UtkarshRedd/Negation_handling/blob/main/NegationHandling.py
    def negation_handler(self, sentence):
        temp = int(0)
        for i in range(len(sentence)):
            if sentence[i-1] in ['not',"n't"]:
                antonyms = []
                for syn in wordnet.synsets(sentence[i]):
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    temp = 0
                    for l in syn.lemmas():
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                    max_dissimilarity = 0
                    for ant in antonyms:
                        syns = wordnet.synsets(ant)
                        w2 = syns[0].name()
                        syns = wordnet.synsets(sentence[i])
                        w1 = syns[0].name()
                        word1 = wordnet.synset(w1)
                        word2 = wordnet.synset(w2)
                        if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                            temp = 1 - word1.wup_similarity(word2)
                        if temp>max_dissimilarity:
                            max_dissimilarity = temp
                            antonym_max = ant
                            sentence[i] = antonym_max
                            sentence[i-1] = ''
        while '' in sentence:
            sentence.remove('')
        return sentence

    def fit(self, X):
        X = X.copy()
        X = self.clean_tweet(X)                     # clean tweets
        X = self.lemmatize(X)                       # lemmatize
        print('Fitting vectorizer...')
        self.vectorizer.fit(X)
        print('Done')

    def transform(self, X):
        X = X.copy()
        # print('Removing Nans...')
        # X = X[~X.isnull()]                          # delete nans
        # X = X[~X.duplicated()]                      # delete duplicates
        
        # X = self.remove_stopwords(X)                 # remove stopwords
        X = self.clean_tweet(X)
        X = self.lemmatize(X)                        # lemmatize

        print('Vectorizing...')
        X = self.vectorizer.transform(X)             # vectorize
        print('Done')
        return X