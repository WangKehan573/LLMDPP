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
sh flan-t5-base.sh
```


## Data sampling

Sample 50 logs from Mac dataset
```shell
python data_sampling --project "Mac" \
                --shot 50
```
python dpp_tf.py --project "Mac" \
                --shot 50

## Fine-tune and Inference

Flan-T5-base or Flan-T5-small (fine-tuned with 50 shot)
```shell
cd flan-t5
python train.py --model "flan-t5-base"\
                --num_epochs 30 \
                --learning_rate 5e-4 \
                --train_percentage "cross" \
                --validation "validation" \
                --systems "Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark"
```

LLaMA (fine-tuned with 50 shot)
```shell
cd llama
sh run.sh 0.025
```

ChatGLM (fine-tuned with 50 shot)
```shell
cd chatglm
sh run.sh 0.025
```

## Evaluation

Evaluate LLM parsing result on certain training dataset size (Flan-T5-base result on 50 shots)
```shell
cd evaluate
python evaluator.py --model "flan-t5-base" \
    --train_percentage "0.025" \
    --systems "Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark" 
```

## Evaluation Results
### RQ1: What is the accuracy of LLM?
<p align="center"><img src="docs/tab2.png" width="800"></p>

### RQ2: How does the accuracy of log parsing vary under different shot sizes?
<p align="center"><img src="docs/tab3.png" width="800"></p>
<p align="center"><img src="docs/tab6.png" width="500"></p>

### RQ3: How is the generalizability of LLMParsers on unseen log templates?
<p align="center"><img src="docs/tab4.png" width="800"></p>

### RQ4: Can pre-trained LLMParsers help improve parsing accuracy?
<p align="center"><img src="docs/tab5.png" width="800"></p>



python dpp_tf.py --project "Mac"   --shot 50
