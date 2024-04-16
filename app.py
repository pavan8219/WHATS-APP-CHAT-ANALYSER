import streamlit as st
import preprocessor
import helper
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect

st.sidebar.title("WhatsApp Chat Analyzer")
st.header('Overall Analysis:-')
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
    if st.sidebar.button("Show Analysis"):
        num_messages, words, num_media_messages, num_links, emojis_count, sentiment_df = helper.fetch_stats(
            selected_user, df)

        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            st.header("Total Messages")
            st.title(num_messages)
        with row1_col2:
            st.header("Total Words")
            st.title(words)
        with row1_col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.header("Links")
            st.title(num_links)

        st.header("Sentiment Analysis")

        st.dataframe(sentiment_df)

        accuracy = helper.perform_sentiment_analysis(sentiment_df)
        st.write(f"Sentiment Analysis Accuracy: {accuracy}")

        # Display visualizations related to overall analysis
        st.title('Most Busy users and their message%')
        x, new_df = helper.most_busy_users(df)
        fig, ax = plt.subplots()

        col1, col2 = st.columns(2)
        with col1:
            ax.bar(x.index, x.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.dataframe(new_df)

        # Word cloud for overall analysis
        st.title('Word Cloud')
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most common words for overall analysis
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emoji analysis for overall analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)
