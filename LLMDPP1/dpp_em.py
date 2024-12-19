import re
import random
import numpy as np
import pandas as pd
import os
import argparse
import json
import time
import math

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


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(selected_item)
    return selected_items


def getDppIndex(log_emb_list,
                item_size,    # log dataset size=2000
                split_ratio):

    max_length = int(item_size * split_ratio)
    feature_vectors = np.array(log_emb_list)

    # standarization no need for log embeddings
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)

    # calculate similarity matrix of log embeddings
    similarities = np.dot(feature_vectors, feature_vectors.T)

    t = time.time()
    result = dpp(similarities, max_length)
    result.sort()
    print('DPP algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
    return result


def DPPsplit(log_list, groundtruth_template, candidate_idx):
    cand_logs = [log_list[idx] for idx in candidate_idx]
    cand_templates = [groundtruth_template[idx] for idx in candidate_idx]
    test_idx = []
    for i in range(len(log_list)):
      if i not in candidate_idx: test_idx.append(i)
    test_idx.sort()
    test_logs = [log_list[idx] for idx in test_idx]
    test_templates = [groundtruth_template[idx] for idx in test_idx]
    return test_logs, cand_logs, test_templates, cand_templates

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

    #Convert DataFrame to list
    content_list = raw_dataset['Content'].values.tolist()
   #for i in range(len(content_list)):
   #     print(i,content_list[i])
    template_list = raw_dataset['EventTemplate'].values.tolist()




    #sampled_log仅仅是编号
    # label result

    file = open(r'./embeddings/Mac.json', "r")
    emb_map = json.load(file)
    #for k in emb_map:
    #    print(k)
    file.close()
    log_embs = []
    i = 0
    for log in content_list:
        #print(i,log)
        i+=1
        log_embs.append(emb_map[log])
    print(f"length of log embs is {len(log_embs)}")
    sampled_log = getDppIndex(log_embs, 2000, precentage)  #candidate_idx是一个列表
    print(sampled_log)
   # python dpp_tf.py --project "Mac" --shot 50

    log_test, log_cand, gt_test, gt_cand = DPPsplit(content_list, template_list, sampled_log)
    data = {'input': log_cand,
            'output': gt_cand}
    res_df = pd.DataFrame(data)
    res_df.insert(0, 'instruction', "Parse the input log to log template.")
    save_path = "./logs/dpp_em/" + project + "/" +"dpp_em"+ str(precentage) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    res_df.to_json(save_path + "train.json", orient="records")


train_data_sample(project=project, precentage=precentage)
