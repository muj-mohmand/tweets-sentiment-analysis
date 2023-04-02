from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd
import torch
import matplotlib.pyplot as plt


list_of_tweets = []
df = pd.read_csv('lakers_tweets.csv')

# iterate over the rows of the DataFrame
for index, row in df.iterrows():
    # process the data in each row
    tweet_words = []
    tweet_proc = ''
    for word in row['Tweet'].split(' '):
        # starts with a mention
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)
    list_of_tweets.append(tweet_proc)


# load model and tokenizer
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

list_of_scores = []

for tweet in list_of_tweets[:100]:
    encoded_input = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    list_of_scores.append(scores)


data = np.array(list_of_scores)
data_mean = np.mean(data, axis=0)

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(['Negative', 'Neutral', 'Positive'], data_mean, color=['r', 'g', 'b'])

# Add axis labels and title
ax.set_xlabel('Sentiment')
ax.set_ylabel('Mean Probability')
ax.set_title('Sentiment Analysis of Lakers Tweets')

# Show the chart
plt.show()
