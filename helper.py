import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('vader_lexicon')

extract = URLExtract()
sid = SentimentIntensityAnalyzer()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    emojis_count = 0
    sentiment_scores = []  # Store sentiment scores for each message

    for message in df['message']:
        # Sentiment analysis for each message
        compound_sentiment = sid.polarity_scores(message)['compound']
        sentiment_scores.append(compound_sentiment)

    # Create a DataFrame with message and corresponding sentiment score
    sentiment_df = pd.DataFrame({'Message': df['message'], 'Sentiment_Score': sentiment_scores})

    sentiment_df['Sentiment_Label'] = sentiment_df['Sentiment_Score'].apply(
        lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral'))

    return num_messages, len(words), num_media_messages, len(links), emojis_count, sentiment_df


def most_busy_users(df):
    # Exclude group notifications and count occurrences of individual users
    filtered_df = df[~df['user'].str.startswith('Group Notification:')].copy()

    # Count occurrences of users and get the top 10
    top_users = filtered_df['user'].value_counts().head(10)

    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})

    return top_users, df


def create_wordcloud(selected_user, df):
    f = open('stopwords.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df['message'] = df['message'].apply(remove_stop_words)
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    f = open('stopwords.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    words = []

    for message in df['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.emoji_count(c) > 0])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def perform_sentiment_analysis(sentiment_df):
    X = sentiment_df['Message']
    y = sentiment_df['Sentiment_Label']

    # Check if there are sufficient samples for splitting
    if len(X) <= 1:
        return "Insufficient data for analysis"

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Split the data into training and testing sets
    if len(X) == 2:
        # If there are only 2 samples, set test_size=0.5 to leave one sample for each set
        X_train, X_test, y_train, y_test = X_vectorized[0], X_vectorized[1], y[0], y[1]
    else:
        # Otherwise, perform standard train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.5, random_state=42)

    # Initialize the Naive Bayes classifier
    nb_classifier = MultinomialNB()

    # Train the classifier
    nb_classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred = nb_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def predict_sentiment(unseen_data):
    sentiment_df = pd.read_csv('sentiment_df.csv')  # Load your sentiment dataframe here

    X = sentiment_df['Message']
    y = sentiment_df['Sentiment_Label']

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Initialize the Naive Bayes classifier
    nb_classifier = MultinomialNB()

    # Train the classifier
    nb_classifier.fit(X_vectorized, y)

    # Transform unseen data into a feature vector
    unseen_data_vectorized = vectorizer.transform([unseen_data])

    # Predict sentiment for unseen data
    predicted_sentiment = nb_classifier.predict(unseen_data_vectorized)
    return predicted_sentiment[0]

# You can add the rest of your code here
