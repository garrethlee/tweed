# Tweed
A naive-bayes classifier for tweets

[Access the website here](https://tweed-app.herokuapp.com)

The website has two main functionalities:

**1. Text Sentiment Analyzer 💬** <br>
Type out text and see a live graphical representation of the text's sentiment (categorized between positive and negative). The graph updates for every key entered.

**2. Twitter User Sentiment Analyzer 🦜** <br>
Enter a Twitter username and analyze their profile as a whole! See a graph representing a user's distribution of negative vs positive tweets, as well as for individual words.

---
**Some drawbacks:**
- The naive-bayes classifier has some trouble classifying negations and double negations (*not happy* is classified as being positive, instead of negative)
- Static nature of model causes it to be unable to adaptively learn
