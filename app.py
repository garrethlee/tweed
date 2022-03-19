import random
from dash import Dash, dcc, html, Input, Output, callback_context, State
from dash.exceptions import PreventUpdate
from main import get_text_sentiment, get_tweets
import plotly.express as px
import pandas as pd
from plots import make_plots

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

def get_positive_message(score):
    if score > 90:
        return "Someone's definitely smiling!"
    elif score > 80:
        return "Awesome!"
    elif score > 70:
        return "Joyful, indeed"
    else:
        return "That's great!"

def get_negative_message(score):
    if score > 90:
        return "Are you okay?"
    elif score > 80:
        return "Cheer up!"
    elif score > 70:
        return "That isn't nice!"
    else:
        return "Calm down, okay."

def make_app_layout(left=[], right=[]):
    return [html.Div(left, className="left-block", style={"width":"49%", "display":"inline"}), 
     html.Div(right, className = "right-block", style={"width":"45%", "display":"inline"})]

def get_try():
    return random.choice(["Im literally dying right now", "That trip was awesome!", 
                    "My dog is on fire", "How not to procrastinate", 
                    "Why does everything suck", "My toes hurt"])

# 'jumpstart' the model for initial loading
cur_positive, cur_negative = get_text_sentiment("?")   

left_block_default = [
    html.H1("Tweed", id="header", className="main-title"),
    html.H2([html.Span("a naive-bayes sentiment analyzer for tweets*", id="subheader-span")], className='subheader'),
    html.H4(id='find-user'),
    html.H4(id='input'),
    html.Button(id='analyze-text', className="submit-btn", children="Analyze Text", n_clicks=0),
    html.Button(id='analyze-user', className="submit-btn", children="Analyze Twitter User", n_clicks=0),   
]

right_block_default = []

left_block_text_analyzer = [
        html.H1("Tweed", id="header", className="main-title"),
        html.H2([html.Span("a naive-bayes sentiment analyzer for tweets*", id="subheader-span")], className='subheader'),
        html.H4(id='find-user'),
        html.Button(id='analyze-text', className="submit-btn", children="Analyze Text", n_clicks=0),
        html.Button(id='analyze-user', className="submit-btn", children="Analyze Twitter User", n_clicks=0),
        html.Div([
            dcc.Input(id="input", placeholder= f'Try "{get_try()}"...', className="tweetBox", value=""),
        ], className = "search-user-div"),
        html.Div(
            [html.H4("How do I interpret the score?", style={"weight":"bold", "text-decoration":"underline", "color":"darkblue"}),
             html.H4("""The score is essentially the probability of the word having a positive sentiment (calculated by the model). 
                        For example, entering 'Friday' will output a negative value of -0.38 and a positive value of 0.62. This means that the model
                        predicts that Friday has a 62% chance being a 'positive' word, and a 38% chance of being a negative word. Since the positive 
                        probability is higher, the model deduces that this is a 'positive' word.""", style={"text-align":"justify"})]
        )]

right_block_text_analyzer = [
        html.H4("Your tweet is: Neutral :0", id="text-output"),
        dcc.Graph(figure={
        "data":[{"y": ["Positive", "Negative"], "x":[cur_positive, cur_negative], "type":"bar" }],
        "layout":{
            "title":"Sentiment Scores"
        }}, id="graph")]

left_block_user_analyzer = [
        html.H1("Tweed", id="header", className="main-title"),
        html.H2([html.Span("a naive-bayes sentiment analyzer for tweets*", id="subheader-span")], className='subheader'),
        html.Button(id='analyze-text', className="submit-btn", children="Analyze Text", n_clicks=0),
        html.Button(id='analyze-user', className="submit-btn", children="Analyze Twitter User", n_clicks=0),
        html.Div([
            dcc.Input(id="input", placeholder="Enter Twitter Username", className="tweetBox", value=""),
            html.Button(id='find-user', className="submit-btn", children="Search", n_clicks=0)
        ], className = "search-user-div", id="search-user-div"),
        html.Div(
            [html.H4("How do I interpret the score?", style={"weight":"bold", "text-decoration":"underline", "color":"darkblue"}),
             html.H4("""The score is essentially the probability of the word having a positive sentiment (calculated by the model). 
                        For example, entering 'Friday' will output a negative score of -0.38 and a positive score of 0.62. This means that the model
                        predicts that Friday has a 62% chance being a 'positive' word, and a 38% chance of being a negative word. Since the positive 
                        probability is higher, the model deduces that this is a 'positive' word.""")]
        )]

app.layout = html.Div(
    [html.Meta(),
    html.Div(left_block_default, id='left-block', className="left-block", style={"width":"49%", "display":"inline"}), 
     html.Div(right_block_default, id='right-block', className = "right-block", style={"width":"45%", "display":"inline-block"})],
      id="app", style={"display":"flex"})

@app.callback(
    Output('app','children'),
    Input('analyze-text','n_clicks'),
    Input('analyze-user','n_clicks'),
    Input('find-user', 'n_clicks'),
    State('input', 'value')
    )
def get_analyze_text(clicks1, clicks2, clicks3, username):
    ctx = callback_context
    if not ctx.triggered:
        return make_app_layout(left_block_default, right_block_default)
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "analyze-text":
            return make_app_layout(left_block_text_analyzer, right_block_text_analyzer) if clicks1 else make_app_layout(left_block_default)
        if button_id == "analyze-user":
            return make_app_layout(left_block_user_analyzer, right_block_default) if clicks2 else make_app_layout(left_block_default)
        if button_id == "find-user":
            tweets = get_tweets(username)
            if not tweets:
                return make_app_layout(left_block_user_analyzer, [html.H3("Username not found!")])
            scores = list(map(get_text_sentiment, tweets))
            data = list(zip(tweets,scores))
            fig1, fig2 = make_plots(data)
            return make_app_layout(left_block_user_analyzer, [dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)]) if clicks3 else make_app_layout(left_block_default)
        

@app.callback(
    Output('graph', 'figure'),
    Input('input', 'value'),
)
def update_graph(value):
    if value is None:
        raise PreventUpdate
    positive, negative = get_text_sentiment(value)
    df = pd.DataFrame({"Sentiment": ["Positive", "Negative"], "Score": [round(positive, 2), -round(negative, 2)]})
    fig = px.bar(df, y="Sentiment", x="Score", text="Score")
    fig.update_layout(title_text=value, title_x=0.5)
    return fig

@app.callback(
    Output('text-output', 'children'),
    Input('input', 'value'),
)
def update_title(value):
    if value is None:
        raise PreventUpdate
    cur_positive, cur_negative = map(lambda x: round(x, 2), get_text_sentiment(value))
    if value == "":
        return f"Your tweet is: Neutral :0"
    if cur_positive > cur_negative:
        msg = get_positive_message(cur_positive*100)
        return f"Your tweet is: Positive :) with {round(cur_positive*100,2)}% confidence. {msg}" 
    if cur_positive < cur_negative:
        msg = get_negative_message(cur_negative*100)
        return f"Your tweet is: Negative :( with {round(cur_negative*100,2)}% confidence. {msg}"
    else:
        return f"Your tweet is: Neutral :0"

if __name__ == "__main__":
    app.run_server(debug=True)