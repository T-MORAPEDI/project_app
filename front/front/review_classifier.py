from textblob import TextBlob

def review_classify(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    return sentiment