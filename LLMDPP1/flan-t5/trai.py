import os
import torch
import argparse
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from openprompt.plms import load_plm
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate
from transformers import T5TokenizerFast

# 设置随机种子
random.seed(41)

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--train_percentage", type=str, default=0.025)
    parser.add_argument("--model", type=str, default="flan-t5-base")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--systems", type=str, default="Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark")
    parser.add_argument("--validation", type=str, default="validation")
    return parser.parse_args()

# 加载并处理数据集
def load_and_process_data(project, percentage, dataset_type):
    dataset_path = f"../logs/{project}/{percentage}/{dataset_type}.json"
    raw_dataset = pd.read_json(dataset_path)
    raw_dataset = raw_dataset.drop(columns=['instruction']).applymap(str)
    column_map = {'input': 'Content', 'output': 'EventTemplate'}
    raw_dataset.rename(columns=column_map, inplace=True)
    return Dataset.from_dict(raw_dataset)

# 准备数据集
def prepare_data(project, percentage):
    datasets = {
        type_: load_and_process_data(project, percentage, type_)
        for type_ in ['train', 'validation', 'test']
    }
    return datasets

# 主程序
def main():
    args = parse_arguments()
    project_list = args.systems.split(",")

    for project in project_list:
        start_time = datetime.now()
        datasets = prepare_data(project, args.train_percentage)

        # 加载预训练模型和分词器
        plm, tokenizer, _, WrapperClass = load_plm(args.model, f"../LLMs/{args.model}/")
        tokenizer = T5TokenizerFast.from_pretrained(f"../LLMs/{args.model}/")
        tokenizer.add_tokens(["<*>", "{", "}", "<", "."])

        # 定义模板和分类器
        template_text = 'Parse the raw log to log template: {"placeholder":"text_a"}  {"mask"}'
        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
        label_words = {0: ["{'meta':'EventTemplate'}"]}
        myverbalizer = GenerationVerbalizer(tokenizer, classes=None, is_rule=True, label_words=label_words)

        # 创建PromptForGeneration模型
        prompt_model = PromptForGeneration(plm=plm, template=mytemplate, tokenizer=tokenizer)

        # 创建DataLoader
        train_dataloader = PromptDataLoader(
            dataset=datasets['train'],
            template=mytemplate,
            verbalizer=myverbalizer,
            tokenizer=tokenizer,
            max_seq_length=256,
            batch_size=args.batch_size,
            shuffle=True
        )

        # 定义优化器和学习率调度器
        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        num_training_steps = args.num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        # 训练循环
        for epoch in range(args.num_epochs):
            prompt_model.train()
            for step, inputs in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                loss = prompt_model(inputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # 保存模型
        output_dir = f"../fine_tuned_model/{args.model}/{project}/{args.train_percentage}/"
        os.makedirs(output_dir, exist_ok=True)
        plm.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        finish_time = datetime.now()
        duration = finish_time - start_time
        print(f"..Running: {duration.days} days, {duration.seconds // 3600} hours, {(duration.seconds // 60) % 60} mins, {duration.seconds % 60} sec.")

if __name__ == "__main__":
    main()
