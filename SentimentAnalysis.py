# import libraries
import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


# download nltk corpus (first time only)
import nltk

nltk.download('all')




# Load the amazon review dataset

df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

df

def preprocess_text(text):

 tokens = word_tokenize(text.lower())

 filtered_token = [token for token in tokens if token not in stopwords.words('english')]

 lemmatizer = WordNetLemmatizer()

 lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_token]

 processed_text = ' '.join(lemmatized_tokens)

 return processed_text

df['reviewText'] = df['reviewText'].apply(preprocess_text)
df

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):

 scores = analyzer.polarity_scores(text)
 sentiment = 1 if scores['pos'] > 0 else 0
 return sentiment

df['sentiment'] = df['reviewText'].apply(get_sentiment_scores)
df.head()

from sklearn.metrics import confusion_matrix

print(confusion_matrix(df['Positive'], df['sentiment']))

from sklearn.metrics import classification_report

print(classification_report(df['Positive'],df['sentiment']))
