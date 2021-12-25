import tweepy 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

## api keys credientials
consumer_key = '1pf3PjmFVrhZENujDIW3GDYJa'
consumer_secret = 'cCbywGr55gngH2ZZmjkDnG1odslsgIENfw9Vklk5VDkBMSneYS'
access_token = '1291153098644365314-emL7GIdGySsl1pY9eyHXUvEIMfxNzi'
access_token_secret = 'MBOcvmxWnFcDw7QtIn67TaG5M9e5SMoN9wem4WmIg6hzu'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

st.title('Tweets Analysis')
st.markdown("""Using the Twitter API, this web app extracts and analyses tweets that contains the keyword(s) you enter in the search box below.""")

## parameters for the tweets
query = st.text_input('Enter keyword')
number_of_tweets = 2500
user_country = []
user_name = []
user_screen_name = []
user_follower_count = []
user_location = []
user_friends_count = []
tweet = []
tweet_likes = []
created_at = []
retweet_count = []

if st.button('Get Tweets') == True:
        
    ## search and display tweets based on parameters entered
    for i in api.search_tweets(q=query,  tweet_mode='extended', count=number_of_tweets):
        user_country.append(i.user.location)
        user_name.append(i.user.name)
        user_screen_name.append(i.user.screen_name)
        user_follower_count.append(i.user.followers_count)
        user_location.append(i.user.location)
        user_friends_count.append(i.user.friends_count)
        tweet.append(i.full_text)
        tweet_likes.append(i.favorite_count)
        created_at.append(i.created_at)
        retweet_count.append(i.retweet_count)

    data = {'created_at':created_at,
                    'user_name':user_name, 
                    'user_screen_name':user_screen_name,
                    'user_followers_count':user_follower_count,
                    'user_friends_count':user_friends_count,
                    'user_location':user_location,
                    'tweet':tweet,'tweet_likes': tweet_likes, 
                    'retweet_count':retweet_count}

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.transpose()
    df = df[~df.tweet.str.contains('RT')]


    ## importing langdetect to determine english tweets
    from langdetect import detect
    ## creating a function for determining english or non-english tweets
    from langdetect import DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0

    def is_english(text):
        try:
            if detect(text) != "en":
                return False
        except LangDetectException:
            return False
        return True

    ## creating new column that contains true or false, if tweet is english or not
    df['is_en'] = df['tweet'].apply(is_english)

    ## remove non-english langs
    df = df[df['is_en'] == True]
    df = df.drop('is_en', axis=1)

    ## dropping values that are not country or state
    df = df[df.user_location.str.contains(',')]

    ## dropping the null values
    df = df.dropna(subset=['user_location'])

    ## reset index
    df = df.reset_index(drop=True)

    ## creating a new column containing the length of each tweet
    df['tweet_len'] = df['tweet'].apply(len)
    st.text('All tweets')
    st.table(df['tweet'])


    st.text('Top locations')

    plt.figure(figsize=(10, 6), dpi=100)
    df['user_location'].value_counts().plot(kind='barh')
    plt.xticks(rotation=90)
    st.pyplot(plt)

    st.text('Top users')

    plt.figure(figsize=(10, 6), dpi=100)
    df['user_name'].value_counts().plot(kind='barh')
    plt.xticks(rotation=90)
    st.pyplot(plt)


    ## col containing raw tweets before cleaning for sentimental analysis using a pretrained model
    df['raw_tweet'] = df['tweet']

    ## text processing function
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords 
    import string


    def text_process(mess):
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation]

        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        
        # Now just remove any stopwords
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


    ## applying the function to our tweet column
    df['tweet'] = df['tweet'].apply(text_process)


    ## ploting top 20 words in the dataframe
    words = df['tweet'].sum()
    from nltk.probability import FreqDist

    fdist = FreqDist()
    for word in df['tweet'].sum():
        fdist[word] += 1

    # fdist
    ## Creating FreqDist for whole BoW, keeping the 20 most common tokens
    all_fdist = FreqDist( fdist).most_common(20)

    ## Conversion to Pandas series via Python Dictionary for easier plotting
    all_fdist = pd.Series(dict(all_fdist))

    ## Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(15,6))

    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax, color="b")
    plt.xticks(rotation=30);
    st.text('Top 20 most common words')

    st.pyplot(plt)


    ## Sentiment Analysis

    ## importing a pretrained model -vadar
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    ## function that calculates the polarity scores of all tweets in each column.
    def get_sentiment(row, **kwargs):
        sentiment_score = sid.polarity_scores(row)
        positive_meter = round((sentiment_score['pos'] * 10), 2)
        negative_meter = round((sentiment_score['neg'] * 10), 2)

        return positive_meter if kwargs['k'] == 'positive' else negative_meter

    ## creating a new column that contains the score of each tweet
    df['positive'] = df['raw_tweet'].apply(get_sentiment, k='positive')
    df['negative'] = df['raw_tweet'].apply(get_sentiment, k='negative')


    for index, row in df.iterrows(): 
        print("Positive : {}, Negative : {}".format(row['positive'], row['negative']))

    st.text('Tweets sentiment overtime')

    ## ploting sent
    df_sent = df[['created_at', 'positive', 'negative']]
    st.line_chart(df_sent.set_index('created_at'))
