from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def lemmatize_token(t:list) -> list:
    """Strips down words in a tweet into its simplest grammatical form"""
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(t):
        if tag[:2] in ("NN", "PRP"):
            pos = "n"
        elif tag[0] == "V":
            pos = "v"
        else:
            pos = "a"
        lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), pos))
    return lemmatized_sentence

def clean_token(token:list) -> list:
    """Removes stopwords from the given token"""
    return [word.lower() for word in token if word.lower() not in stop_words]

def tweet_pipeline(tweet: str) -> dict:
    """Pipeline for transforming a tweet in raw text format by
        1. Tokenizing the tweet (Dividing the tweet into words/subparts)
        2. Lemmatizing the tweet (stripping to its simplest grammatical form)
        3. Cleaning the tweet (removing noise, symbols, stop words, etc.)
        
        Returns: a dictionary with each remaining token"""
    token = word_tokenize(tweet)
    lemmatized_token = lemmatize_token(token)
    cleaned_token = clean_token(lemmatized_token)
    return dict((t.lower(),True) for t in cleaned_token)

def load_model():
    """Loads model from pickle file"""
    with open("naivebayes_model.h5", "rb") as f:
        model = pickle.load(f)
        return model
        
