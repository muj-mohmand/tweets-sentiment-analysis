import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime


todays_date = datetime.today().strftime("%Y-%m-%d")

query = "lakers until:{} since:2023-02-11".format(todays_date)
print(query)
tweets = []
limit = 10000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.rawContent])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

pd.options.display.max_rows = 10000
df.to_csv('lakers_tweets.csv', index=False)
