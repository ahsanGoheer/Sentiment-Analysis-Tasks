import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random


nltk.download('twitter_samples')

all_positive_tweets=twitter_samples.strings('positive_tweets.json')
all_negative_tweets=twitter_samples.strings('negative_tweets.json')



print("Number of positive tweets : ",len(all_positive_tweets))
print("Number of negative tweets : ",len(all_negative_tweets))

figure= plt.figure(figsize=(10,10))
labels=['positive','negative']

sizes=[len(all_positive_tweets),len(all_negative_tweets)]

plt.pie(sizes,labels=labels,autopct="%1.1f%%",shadow=True,startangle=90)

plt.axis('equal')

plt.show()

print("Positive tweets :")
print(all_positive_tweets[0:10])
print("Negative tweets :")
print(all_negative_tweets[0:10])

nltk.download('stopwords')


import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


sample_tweet= all_positive_tweets[2000]
print("Sample Tweet : ",sample_tweet)

cleaned_sample=re.sub(r'^RT[\s]+','',sample_tweet)
print("Cleaned Tweet : ",cleaned_sample)

cleaned_sample=re.sub(r'https?:\/\/.*[\r\n]*','',cleaned_sample)
print("Further Cleaned : ",cleaned_sample)

cleaned_sample=re.sub(r'#','',cleaned_sample)
print("Removed Hashtag : ",cleaned_sample)



tokenizer=TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)

tweet_tokens=tokenizer.tokenize(cleaned_sample)
print('Tokenized String : ')
print(tweet_tokens)

stop_words=stopwords.words('english')

print('Stop Words \n')
print(stop_words)
print('\nPunctuation\n')
print(string.punctuation)

tweet_clean=[]
for word in tweet_tokens:
    if(word not in stop_words and word not in string.punctuation):
        tweet_clean.append(word)


stemmer=PorterStemmer()

tweets_stem=[]

for word in tweet_clean:
    stem_word=stemmer.stem(word)
    tweets_stem.append(stem_word)

print('Stemmed Words:')
print(tweets_stem)


