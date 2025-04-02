# Empirical Analysis of LLMDPP: Advancing Log Parsing in the LLM Era




## Environment 
### Requirement
```shell
sh env_init.sh
```

### Large Language Models

To download the Large Language Models:
```shell
cd LLMs
sh flan-t5-small.sh
```


## Data sampling

Sample 50 logs from Mac dataset
```shell
tf-idf
python dpp_tf.py --project "Mac" \
                --percentage 0.025

embedding:
python embedding.py
python dpp_em.py --project "Mac" \
                --percentage 0.025
```


## Fine-tune and Inference

Flan-T5-small (fine-tuned with 50 shot)
```shell
cd flan-t5
python train.py --model "flan-t5-small"\
                --num_epochs 30 \
                --learning_rate 5e-4 \
                --train_percentage "cross" \
                --validation "validation" \
                --systems "Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark"
```

## Evaluation

Evaluate LLM parsing result on certain training dataset size (Flan-T5-base result on 50 shots)
```shell
cd evaluate
python evaluator.py --model "flan-t5-small" \
    --train_percentage "dpp_tf_0.025" \
    --systems "Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark" 
``


Our code draws on the following projects：

https://github.com/logpai/logparser

https://github.com/zeyang919/LLMParser


