from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import emoji
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

# def emoji_helper(selected_user,df):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#
#     emojis = []
#     for message in df['message']:
#         emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])
#
#     emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
#
#
#     return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def percentage(df,k):
    df = round((df['user'][df['value']==k].value_counts() / df[df['value']==k].shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df


# Text preprocessing for user similarities
def preprocess_text(text):
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert text to lowercase
    text = text.lower()

    # Tokenization and remove stopwords
    stopwords_set = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords_set]

    # Join the tokens back into a cleaned message
    cleaned_text = " ".join(tokens)

    return cleaned_text


def similar_user(df):
    df = df.groupby('user')['message'].apply(' '.join).reset_index()
    # Apply text preprocessing to the messages
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_message'])

    # Calculate pairwise cosine similarity between users
    user_similarity_matrix = cosine_similarity(tfidf_matrix)

    # Create a DataFrame with 'User' as index and columns for most similar user and their similarity score
    user_names = df['user'].unique()
    similarity_df = pd.DataFrame(index=user_names, columns=['Most Similar User', 'Similarity Score'])

    # Find the most similar user for each user
    for user in user_names:
        user_index = df[df['user'] == user].index[0]
        similarities = user_similarity_matrix[user_index]
        similarities[user_index] = -1  # Set similarity to -1 for the user itself to exclude it
        most_similar_user_index = similarities.argmax()
        most_similar_user = df.loc[most_similar_user_index, 'user']
        similarity_score = similarities.max()
        similarity_df.at[user, 'Most Similar User'] = most_similar_user
        similarity_df.at[user, 'Similarity Score'] = similarity_score

    return similarity_df

















