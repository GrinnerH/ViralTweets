import pandas as pd

# 使用绝对路径读取 Parquet 文件
df = pd.read_parquet("/home/robot/wwh/ViralTweets/classification/model_with_extra_features/final_dataset_since_october_2022.parquet.gzip")
df.to_csv("all_tweets.csv")
# 提取 author_id 列并去重
author_ids = df['author_id'].drop_duplicates()

# 打印结果
print(author_ids)

# 保存到 CSV
author_ids.to_csv("users_ids.csv", index=False)
