import pandas as pd

# 使用绝对路径读取 Parquet 文件
df = pd.read_parquet("/home/robot/wwh/ViralTweets/classification/model_with_extra_features/final_dataset_since_october_2022.parquet.gzip")
#df.to_csv("all_tweets.csv")

# 提取users相关列
# users_df = final_df[['author_id', 'followers_count', 'following_count', 'tweet_count', 'protected', 'verified', 'username']]

# 去重，确保每个author_id只出现一次
# users_df = users_df.drop_duplicates(subset='author_id')

# 保存为users.parquet.gzip
# users_df.to_parquet('users.parquet.gzip', compression='gzip')

# 提取all_tweets相关列
tweets_columns = ['text', 'possibly_sensitive', 'lang', 'created_at', 'id', 'author_id', 'retweet_count', 'reply_count',
                  'like_count', 'quote_count', 'has_media', 'topic_domains', 'topic_entities', 'hashtags', 'urls', 'viral',
                   'tweet_length', 'sentiment', 'sentiment_score', 'nb_of_hashtags',
                  'mentions', 'nb_of_mentions']

all_tweets_df = df[tweets_columns]

# 保存为all_tweets.parquet.gzip
all_tweets_df.to_parquet('all_tweets.parquet.gzip', compression='gzip')


# df.to_excel('all_tweets_with_features.xlsx', index=False)
