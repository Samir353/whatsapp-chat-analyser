import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter

st.sidebar.title("Whatsapp Chat Analyzer")

nltk.download('vader_lexicon')

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)



    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Object
    sentiments = SentimentIntensityAnalyzer()

    # Creating different columns for (Positive/Negative/Neutral)
    df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
    df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
    df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]  # Neutral


    # To indentify true sentiment per row in message column
    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0


    # Creating new column & Applying function
    df['value'] = df.apply(lambda row: sentiment(row), axis=1)

    st.title("WhatsApp Chat Dataframe")
    st.dataframe(df)

    # fetch unique users and sorting them
    user_list_1 = df['user'].unique().tolist()
    user_counts = Counter(df['user'])
    user_list = sorted(user_list_1, key=lambda user: user_counts[user], reverse=True)

    try:
        user_list.remove('group_notification')
    except ValueError:
        pass
    # user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        unique_users_count = df['user'].nunique()
        st.title("Top Statistics")
        col0, col1, col2, col3, col4 = st.columns(5)

        with col0:
            st.header("Total User")
            st.title(f'{unique_users_count-1},')

        with col1:
            st.header("Total Messages")
            st.title(f'{num_messages},')
        with col2:
            st.header("Total Words")
            st.title(f'{words},')
        with col3:
            st.header("Media Shared")
            st.title(f'{num_media_messages},')
        with col4:
            st.header("Links Shared")
            st.title(f'{num_links}')

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df[df['user'] != 'group_notification'])
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)



        # Percentage contributed
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.header("Most positive contributers")
                x = helper.percentage(df[df['user'] != 'group_notification'], 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.header("Most neutral contributers")
                y = helper.percentage(df[df['user'] != 'group_notification'], 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.header("Most negative contributers")
                z = helper.percentage(df[df['user'] != 'group_notification'], -1)

                # Displaying
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.header("Most positive user")

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.header("Most neutral user")

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.header("Most negative user")

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # Overall Sentiment of the group
        if selected_user == 'Overall':
            st.header('Overall Sentiment of the Group')
            pos_score = (df['value'] == 1).sum()
            neg_score = (df['value'] == -1).sum()
            total = pos_score + neg_score
            pos_ratio = (pos_score/total)*100
            neg_ratio= (neg_score/total)*100
            # Create a pie chart
            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie([pos_ratio, neg_ratio], labels=['Positive', 'Negative'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Display the pie chart in Streamlit
            st.pyplot(fig)

            # Display the percentages
            st.write(f'Positive Percentage: {pos_ratio:.2f}%')
            st.write(f'Negative Percentage: {neg_ratio:.2f}%')

        else:
            st.header('Your Sentiment from the Group')
            # Filter the DataFrame based on the specified user
            filtered_df = df[df['user'] == selected_user]
            # Count the number of ones in the 'value' column for the specified user
            pos_score = (filtered_df['value'] == 1).sum()
            neg_score = (filtered_df['value'] == -1).sum()
            total = pos_score + neg_score
            pos_ratio = (pos_score/total)*100
            neg_ratio= (neg_score/total)*100
            # Create a pie chart
            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie([pos_ratio, neg_ratio], labels=['Positive', 'Negative'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Display the pie chart in Streamlit
            st.pyplot(fig)

            # Display the percentages
            st.write(f'Positive Percentage: {pos_ratio:.2f}%')
            st.write(f'Negative Percentage: {neg_ratio:.2f}%')

        # who is similar to whom
        if selected_user == 'Overall':
            st.header('Who is most similar to Whom')
            s_df=helper.similar_user(df)
            s_df = s_df.sort_values(by='Similarity Score', ascending=False)
            # Assuming s_df is your DataFrame
            s_df = s_df[s_df['Similarity Score'] != 1]
            st.dataframe(s_df)
        else:
            st.header('You are most similar to')
            s_df=helper.similar_user(df)
            s_df = s_df.sort_values(by='Similarity Score', ascending=False)
            st.dataframe(s_df.loc[selected_user])








        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

         # emoji analysis
        # emoji_df = helper.emoji_helper(selected_user,df)
        # st.title("Emoji Analysis")
        #
        # col1,col2 = st.columns(2)
        #
        # with col1:
        #     st.dataframe(emoji_df)
        # with col2:
        #     fig,ax = plt.subplots()
        #     ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
        #     st.pyplot(fig)











