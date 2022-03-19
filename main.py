from model import load_model, tweet_pipeline
from auth import get_token, create_url
import requests
import re
import json

classifier = load_model()

def clean_tweets(tweets):
    """Removes whitespace and non-essential characters from scraped tweets"""
    for i in range(len(tweets)):
        tweet = tweets[i]
        tweet = re.sub(r"((www.)?https?:\/\/)?[^\s]*\.([\w]{2,3})(\/\w*)*", "", tweet) #removes links
        tweet = re.sub(r"(RT )?@\w+:?", "", tweet) #removes RT and @
        tweet = re.sub(r"[^(a-zA-Z|\')]", " ", tweet)
        tweet = ' '.join(tweet.split()) #removes whitespace from text
        tweets[i] = tweet
    final_tweets = list(filter(lambda tweet: tweet != "", tweets))
    return final_tweets

def get_tweets(user):
    """Requests tweets from from Twitter API"""
    url = create_url(user)
    bearer_token = get_token()
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers)
    d = json.loads(response.text)
    try:
        tweets = [entry['text'] for entry in d['data']]
        cleaned_tweets = clean_tweets(tweets)
        return cleaned_tweets
    except KeyError as e:
        return None
        
def get_sentiments(user):
    """Calculates latest 10 tweet sentiments for given Twitter user"""
    classifier = load_model()
    tweets = get_tweets(user)
    for tweet in tweets[:10]:
        token = tweet_pipeline(tweet)
        sentiment = classifier.prob_classify(token)
        print(f"Tweet: {tweet}\nNegative: {sentiment.prob('Negative')}\nPositive:{sentiment.prob('Positive')}\n") 

def get_text_sentiment(tweet):
    """Calculates sentiment for individual tweet""" 
    token = tweet_pipeline(tweet)
    sentiment = classifier.prob_classify(token)
    return (sentiment.prob('Positive'), sentiment.prob('Negative'))

# user = input("What username do you wanna check out? ").strip().lower()
# get_sentiments(user)