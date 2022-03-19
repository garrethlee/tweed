from nltk import pos_tag
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle


def get_max_score(v):
    pos, neg = v
    if pos > neg:       
        return pos
    else:
        return -neg

def make_plots(data):
    with open('naivebayes_model.h5', 'rb') as f:
        model = pickle.load(f)
        tweets = []
        tags = []
        scores = []
        counts = {}
        for tweet, score in data:
            tweets.append(tweet)
            scores.append(score)
            for word in tweet.lower().split():
                counts[word] = counts.get(word, 0) + 1
            
        processed_scores = list(map(get_max_score, scores))
        df = {"tweet":[i for i in range(len(data))], "sentiment":processed_scores}
        fig1 = go.Figure(px.bar(x=df["tweet"], y= df["sentiment"], 
                            labels={"x":"Tweets", "y":"Sentiment (Positive / Negative)"}))
        fig1.update_layout(title_text = "Tweet Sentiment Trends in Chronological Order")
        fig1.update_traces(hovertemplate=[f"Tweet: {tweet} ({y} score)" for tweet,y in zip(tweets, df['sentiment'])])

        all_data = {}

        for word in counts:
            pos = model.prob_classify({word:True}).prob('Positive')
            neg = model.prob_classify({word:True}).prob('Negative')   
            all_data[word] = get_max_score((pos,neg))
            tags.append(pos_tag([word], tagset='universal')[0][1])
            
        df = pd.DataFrame({"Word":all_data.keys(), "Frequency":counts.values(), "Sentiment Score":all_data.values(), "Word Type":tags})
        fig2 = px.scatter(df, x="Sentiment Score", y="Frequency", size='Frequency', hover_name="Word", color="Word Type")

        fig2.update_layout(title_text="Word Frequency and Sentiment Scores")

        return fig1, fig2