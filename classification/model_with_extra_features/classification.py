# MODIFY AS REQUIRED
import torch
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

from transformers import get_scheduler

from tqdm.auto import tqdm

import evaluate

from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

from text_preprocessing import clean_tweet, clear_reply_mentions, normalizeTweet
from custom_model import CustomModel

'''
DATA_PATH = "../../data"

PROCESSED_PATH = f"{DATA_PATH}/processed"

PROCESSED_PATH_VIRAL = f'{DATA_PATH}/new/processed/viral'
PROCESSED_PATH_COVID = f'{DATA_PATH}/new/processed/covid'
'''

# Different models
BERT_BASE_UNCASED = "bert-base-uncased"
BERT_BASE_CASED = "bert-base-cased"
ROBERTA_BASE = "roberta-base"
BERT_TWEET = "vinai/bertweet-base"

# TODO: Don't forget to cite papers if you use some model
BERT_TINY = "prajjwal1/bert-tiny"

TWEET_MAX_LENGTH = 280

# TEST SPLIT RATIO + MODELS (ADD MORE MODELS FROM ABOVE)
MODELS = [BERT_TWEET, BERT_TINY, BERT_BASE_CASED, ROBERTA_BASE]
TEST_RATIO = 0.2

TOP_FEATURES = ["verified", "tweet_length", "possibly_sensitive", "sentiment", "nb_of_hashtags", "has_media", "nb_of_mentions"]

def preprocess_data(dataset):
    # 将 has_media 列的数据类型转换为整数（0或1），表示是否包含媒体。
    dataset.loc[:, 'has_media'] = dataset.has_media.astype("int")
    # 将 possibly_sensitive 列的数据类型转换为整数（0或1），表示是否可能包含敏感内容。
    dataset.loc[:, 'possibly_sensitive'] = dataset.possibly_sensitive.astype("int")

    # 过滤出情感置信度分数大于0.7的推文（高情感倾向）。
    #dataset = dataset[dataset.sentiment_score > 0.7]
    # 将情感标签转换为数值（POSITIVE为1，NEGATIVE为0）。
    dataset.loc[:, 'sentiment'] = dataset.sentiment.replace({'POSITIVE': 1, 'NEGATIVE': 0})
    # 将 verified 列的数据类型转换为整数（0或1），表示是否为已验证用户。
    dataset.loc[:, 'verified'] = dataset['verified'].astype(int)

    # remove tweets with 0 retweets (to eliminate their effects) 去除转发数为0的推文，以消除它们的影响
    #dataset = dataset[dataset.retweet_count > 0]

    ## UPDATE: Get tweets tweeted by the same user, on the same day he tweeted a viral tweet

    # Get the date from datetime
    # normalize() sets all datetimes clock to midnight, which is equivalent as keeping only the date part
    # 从 created_at 列提取日期部分，并将时间归一化（设为午夜），只保留日期信息。
    dataset['date'] = dataset.created_at.dt.normalize()

    viral_tweets = dataset[dataset.viral]
    non_viral_tweets = dataset[~dataset.viral]

    # 将非病毒推文与病毒推文合并，基于 author_id 和 date，提取必要的列。
    # 提取该用户发布viral推文的同一天发布的推文，并且添加id_y列和vial_y列，id_y是该日期中发布的那一个viral推文的id，viral_y是True
    temp = non_viral_tweets.merge(viral_tweets[['author_id', 'date', 'id', 'viral']], on=['author_id', 'date'], suffixes=(None, '_y'))

    # 提取病毒推文的ID。
    same_day_viral_ids = temp.id_y.unique()

    # 获取同一天发布的病毒推文，并去除重复项。
    same_day_viral_tweets = viral_tweets[viral_tweets.id.isin(same_day_viral_ids)].drop_duplicates(subset=['author_id', 'date'])
    # 从合并的结果中获取同一天发布的非病毒推文，并去除重复项。
    same_day_non_viral_tweets = temp.drop_duplicates(subset=['author_id', 'date'])

    logging.info(f"Number of viral tweets tweeted on the same day {len(same_day_viral_tweets)}")
    logging.info(f"Number of non viral tweets tweeted on the same day {len(same_day_non_viral_tweets)}")

    # 将同一天的病毒推文和非病毒推文合并成一个新的数据集。
    dataset = pd.concat([same_day_viral_tweets, same_day_non_viral_tweets], axis=0)
    # 选择要保留的列，包括ID、文本、特征和病毒标志。
    dataset = dataset[['id', 'text'] + TOP_FEATURES + ['viral']]

    # Balance classes to have as many viral as non viral ones 平衡数据集中的病毒和非病毒推文的数量。
    #dataset = pd.concat([positives, negatives.sample(n=len(positives))])
    #dataset = pd.concat([positives.iloc[:100], negatives.sample(n=len(positives)).iloc[:200]])

    # Clean text to prepare for tokenization 去除空值推文，以准备后续的文本处理。
    #dataset = dataset.dropna()
    # 将 viral 列的数据类型转换为整数。
    dataset.loc[:, "viral"] = dataset.viral.astype(int)

    # TODO: COMMENT IF YOU WANT TO KEEP TEXT AS IS
    # 对推文文本进行清理，生成一个新列 cleaned_text，用于后续处理，注释提示如果想保留原始文本可以取消清理。
    dataset["cleaned_text"] = dataset.text.apply(lambda x: clean_tweet(x, demojize_emojis=False))

    # 去除任何包含缺失值的行。
    dataset = dataset.dropna()
    # 将特征列的值转换为列表，创建一个新的列 extra_features。
    dataset.loc[:, "extra_features"] = dataset[TOP_FEATURES].values.tolist()
    # 选择最终保留的列，包括ID、清理后的文本、额外特征和病毒标志。
    dataset = dataset[['id', 'cleaned_text', 'extra_features', 'viral']]

    return dataset

def prepare_dataset(sample_data, balance=False):
    # Split the train and test data st each has a fixed proportion of viral tweets
    # 使用 train_test_split 函数将样本数据分为训练集和验证集，确保每个集合中病毒推文的比例相同。
    train_dataset, eval_dataset = train_test_split(sample_data, test_size=TEST_RATIO, random_state=42, stratify=sample_data.viral)

    # Balance test set
    # 如果 balance 为 True，则对验证集进行平衡处理：
    # 筛选出所有病毒推文和非病毒推文。
    # 通过对非病毒推文进行随机抽样，使其数量与病毒推文相等。
    # 将这两个数据集合并，形成平衡后的验证集。
    if balance:
        eval_virals = eval_dataset[eval_dataset.viral == 1]
        eval_non_virals = eval_dataset[eval_dataset.viral == 0]
        eval_dataset = pd.concat([eval_virals, eval_non_virals.sample(n=len(eval_virals))])

    # 记录训练集的信息，包括样本总数、病毒推文数量和非病毒推文数量。
    logging.info('{:>5,} training samples with {:>5,} positives and {:>5,} negatives'.format(
        len(train_dataset), len(train_dataset[train_dataset.viral == 1]), len(train_dataset[train_dataset.viral == 0])))
    #记录验证集的信息，包括样本总数、病毒推文数量和非病毒推文数量。
    logging.info('{:>5,} validation samples with {:>5,} positives and {:>5,} negatives'.format(
        len(eval_dataset), len(eval_dataset[eval_dataset.viral == 1]), len(eval_dataset[eval_dataset.viral == 0])))

    # 将训练集和验证集分别保存为压缩的 Parquet 格式文件，文件名分别为 train.parquet.gzip 和 test.parquet.gzip。
    train_dataset.to_parquet("train.parquet.gzip", compression='gzip')
    eval_dataset.to_parquet("test.parquet.gzip", compression='gzip')

    # 使用 load_dataset 函数加载刚才保存的 Parquet 文件，并将其存储在变量 ds 中。
    ds = load_dataset("parquet", data_files={'train': 'train.parquet.gzip', 'test': 'test.parquet.gzip'})
    return ds

# 将原始文本切分成子单元的过程就叫做Tokenization
def tokenize_function(example, tokenizer):
  # Truncate to max length. Note that a tweet's maximum length is 280
  # TODO: check dynamic padding: https://huggingface.co/course/chapter3/2?fw=pt#dynamic-padding
  return tokenizer(example["cleaned_text"], truncation=True)


def test_all_models(ds, nb_extra_dims, models=MODELS):
    # 初始化一个字典 models_losses，用于存储每个模型的损失记录。
    models_losses = {}
    # 检查是否有可用的 GPU，如果有，则使用 CUDA 设备；否则，使用 CPU。
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 初始化一个空字符串 output，用于存储各模型的评估结果。
    output = ""

    # 遍历每个模型的检查点（即预训练模型的路径）。
    for checkpoint in models:
        # 清空 CUDA 缓存，以释放显存。
        torch.cuda.empty_cache()
        # 使用 AutoTokenizer 从指定的检查点加载 tokenizer。
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # 创建一个 CustomModel 实例，传入模型检查点、额外维度数量和标签数量（这里是二分类）。
        custom_model = CustomModel(checkpoint, num_extra_dims=nb_extra_dims, num_labels=2)
        # 将模型移动到指定的设备（GPU 或 CPU）。
        custom_model.to(device)

        # 对数据集 ds 进行映射，使用 tokenize_function 函数对每个样本进行分词处理，batched=True 表示批量处理。
        tokenized_datasets = ds.map(lambda x: tokenize_function(x, tokenizer=tokenizer), batched=True)
        # 创建一个数据整理器 data_collator，用于对输入进行填充，以确保批次内的样本具有相同的长度。
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # 从分词后的 tokenized_datasets 中移除不需要的列，包括 __index_level_0__、cleaned_text 和 id。
        tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__", "cleaned_text", "id"])
        # 重命名 viral 列为 labels，以符合模型训练的要求。
        tokenized_datasets = tokenized_datasets.rename_column("viral", "labels")
        # 设置数据集的格式为 PyTorch 格式，以便后续使用 PyTorch 的数据加载器。
        tokenized_datasets.set_format("torch")

        # 定义每个批次的大小为 32。
        batch_size = 32

        # 创建训练数据加载器 train_dataloader ，从训练集加载数据，并随机打乱数据。使用 DataLoader 类加载数据。
        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        # 创建验证数据加载器 eval_dataloader，从验证集加载数据。
        eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)

        # 定义损失函数为 BCEWithLogitsLoss，二元交叉熵损失（BCE），适用于二分类问题。
        criterion = BCEWithLogitsLoss()
        # 创建优化器 optimizer，使用 AdamW 优化算法，学习率设为 5e-5。
        optimizer = AdamW(custom_model.parameters(), lr=5e-5)

        # 定义训练的轮数为 15。
        num_epochs = 15
        # 计算总的训练步数，即每轮的训练批次数乘以训练轮数。
        num_training_steps = num_epochs * len(train_dataloader)
        # 创建学习率调度器 lr_scheduler，使用线性调度器，无预热步数，总训练步数为 num_training_steps。
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # 初始化一个进度条 progress_bar，用于显示训练进度。
        progress_bar = tqdm(range(num_training_steps))

        # 初始化一个空列表 losses，用于存储每个训练步骤的损失值。
        losses = []
        # 设置模型为训练模式。
        custom_model.train()
        # 开始训练轮次的循环。
        for epoch in range(num_epochs):
            # 遍历训练数据加载器中的每个批次。
            for batch in train_dataloader:
                # 将批次中的所有数据移到指定的设备（GPU 或 CPU）。这里的 batch 是一个字典，包含输入和标签。
                batch = {k: v.to(device) for k, v in batch.items()}
                # 使用模型对批次数据进行前向传播，获取输出 logits，并使用 squeeze 去除多余的维度。
                logits = custom_model(**batch).squeeze()

                # 计算损失值，将 logits 和实际标签（batch['labels']）传入损失函数。
                loss = criterion(logits, batch['labels'].float())
                # 将当前损失值添加到 losses 列表中（注释掉的行是将损失值移到 CPU 后添加）。
                #losses.append(loss.cpu().item())
                losses.append(loss.item())
                # 执行反向传播，以计算梯度。
                loss.backward()

                # 更新优化器
                optimizer.step()
                # 更新学习率调度器
                lr_scheduler.step()
                # 将优化器的梯度清零，为下一个批次准备。
                optimizer.zero_grad()
                # 更新进度条。
                progress_bar.update(1)
        # 将当前模型的损失记录存入 models_losses 字典。
        models_losses[checkpoint] = losses

        # 初始化一个指标组合 metric，用于计算准确率、召回率、精确率和 F1 分数。
        metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
        # 设置模型为评估模式。
        custom_model.eval()
        # 遍历验证数据加载器中的每个批次。
        for batch in eval_dataloader:
            # 将批次中的数据移到指定设备。
            batch = {k: v.to(device) for k, v in batch.items()}
            # 在不计算梯度的上下文中，执行前向传播以获取 logits。
            with torch.no_grad():
                logits = custom_model(**batch)

            #计算模型的预测结果，将 logits 经过 Sigmoid 激活函数后四舍五入以得到最终预测。
            #predictions = torch.argmax(outputs, dim=-1)
            predictions = torch.round(torch.sigmoid(logits))
            # 将当前批次的预测结果和真实标签添加到指标中，用于后续计算性能。
            metric.add_batch(predictions=predictions, references=batch["labels"])
        # 将当前模型的评估结果格式化为字符串，添加到 output 中。
        output += f"checkpoint: {checkpoint}: {metric.compute()}\n"
    logging.info(output)
    # 将评估结果写入名为 same_day_as_viral_with_features_train_test_balanced_accuracy.txt 的文本文件。
    with open("same_day_as_viral_with_features_train_test_balanced_accuracy.txt", "w") as text_file:
        text_file.write(output)
    # 返回每个模型的损失记录。
    return models_losses

def main():
    # DATA FILE SHOULD BE AT THE ROOT WITH THIS SCRIPT
    all_tweets_labeled = pd.read_parquet(f'final_dataset_since_october_2022.parquet.gzip')

    # 自行添加
    # all_tweets_labeled.to_csv('all_tweets_labeled.csv', index=False)

    dataset = preprocess_data(all_tweets_labeled)
    ds = prepare_dataset(dataset, balance=True)

    nb_extra_dims = len(TOP_FEATURES)
    test_all_models(ds, nb_extra_dims=nb_extra_dims)

if __name__ == "__main__":
    main()