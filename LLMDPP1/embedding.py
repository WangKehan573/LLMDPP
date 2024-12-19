import openai
import json
import os
import pandas as pd
from tqdm import tqdm
from openai.embeddings_utils import get_embedding
import argparse

from transformers import T5Tokenizer, T5Model
import torch

# load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
model = T5Model.from_pretrained(r'C:\Users\wangk\Desktop\flan-t5-small')

# set the model to evaluation mode
model.eval()


# encode the input sentence
def get_embedding(line):
    input_ids = tokenizer.encode(line, return_tensors='pt', max_length=512, truncation=True)
    # generate the vector representation
    with torch.no_grad():
        outputs = model.encoder(input_ids=input_ids)
        vector = outputs.last_hidden_state.mean(dim=1)
    return vector[0]





if os.path.exists("embeddings") == False:
    os.mkdir("embeddings")
input_dir = r".\LLMDPP\logs"
output_dir = "embeddings/"
log_list = ['Mac']#['HDFS', 'Spark', 'BGL', 'Windows', 'Linux', 'Android', 'Mac', 'Hadoop', 'HealthApp', 'OpenSSH', 'Thunderbird', 'Proxifier', 'Apache', 'HPC', 'Zookeeper', 'OpenStack']

for logs in log_list:
    embedding = dict()
    print("Embedding " + logs + "...")
    i = pd.read_csv("logs/" + logs + "/" + logs + "_2k.log_structured_corrected.csv")
    contents = i['Content']
    for log in tqdm(contents):
        response = get_embedding(log)     
        embedding[log] = response
    o = json.dumps(embedding, separators=(',',':'))
    f = open(output_dir + logs + ".json","w")
    f.write(o)
    f.close()