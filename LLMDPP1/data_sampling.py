import re
import random
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="Mac")
parser.add_argument("--percentage", type=str, default="0.025")
parser.add_argument("--shot", type=str, default="0")
args = parser.parse_args()


project=args.project
if args.shot!=0:
    args.shot = int(args.shot)
    precentage=float(args.shot/2000)
else:
    precentage=args.precentage



def replace_numbers_with_zero(text):
    return re.sub(r'\d+(\.\d+)?', '0', text) #将所有数字替换为0


def train_data_sample(project, precentage):
    dataset_path = "logs/" + project + "/" + project + "_2k.log_structured_corrected.csv"
    # load the dataset and make statistics
    keep_columns = ["LineId", "Content", "EventTemplate"]
    raw_dataset = pd.read_csv(dataset_path, index_col=False, usecols=keep_columns)
    # print(raw_dataset)
    raw_dataset = raw_dataset.applymap(str)

    # Extract the text column
    raw_dataset['Content_0'] = raw_dataset['Content'].apply(replace_numbers_with_zero)
    text_column = raw_dataset['Content_0']
    #text_column = raw_dataset['EventTemplate']

    # Text preprocessing and vectorization
    vectorizer = TfidfVectorizer()
    data_matrix = vectorizer.fit_transform(text_column).toarray()

    # Mean Shift clustering
    #scaler = StandardScaler()
    #data_matrix = scaler.fit_transform(data_matrix)
    mean_shift = MeanShift(bandwidth=0.5)
    clusters = mean_shift.fit_predict(data_matrix).tolist()
    unique_labels = set(clusters)  # 去除重复的标签

    cluster_counts = {}

    # 遍历每个样本的簇标签
    for label in clusters:
        if label not in cluster_counts:
            cluster_counts[label] = 0
        cluster_counts[label] += 1

    # 对字典按样本数量从大到小排序
    sorted_clusters = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)

    # 打印每个簇的样本数量
    for label, count in sorted_clusters:
        print(f"Cluster {label}: {count} samples")


    content_list = raw_dataset['Content'].tolist()
    print(len(content_list))
    cluster_dict = {}
    for data, cluster_id in zip(content_list, clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(data)
    sorted_clusters = sorted(cluster_dict.values(), key=len, reverse=True)

    sampled_log = []
    while len(sampled_log) < int(len(raw_dataset) * precentage):
        for i in sorted_clusters:
            if len(sampled_log) == int(len(raw_dataset) * precentage):
                break
            if i != []:
                sample = random.choice(i)
                sampled_log.append(sample)
                i.remove(sample)
    # label result
    template_list = []
    for element in sampled_log:
        value = raw_dataset.loc[raw_dataset["Content"] == element, 'EventTemplate'].values[0]
        template_list.append(value)
    data = {'input': sampled_log,
            'output': template_list}
    res_df = pd.DataFrame(data)
    res_df.insert(0, 'instruction', "Parse the input log to log template.")
    save_path = "./logs/" + project + "/" + str(precentage) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    res_df.to_json(save_path + "train.json", orient="records")


train_data_sample(project=project, precentage=precentage)