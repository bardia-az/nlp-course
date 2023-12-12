# Stock Market Prediction using Textual Data

This repository is the implementation of the final NLP course project of the group savagetokenizer.


## Environment

We recommand to use a Python virtual environment with Python = 3.8.7. The requirements can be installed with:

```
git clone https://csil-git1.cs.sfu.ca/baa27/nlpclass-1237-g-savagetokenizer.git
cd nlpclass-1237-g-savagetokenizer/project
pip install -r requirements.txt
```



## Check the accuracy of the models without the weights

### 0. Download Data

Please download and extract the contents of `output.zip` file into a folder named `models` in the `project` directory.


### 1. Run `check.py` with the appropriate arguments

First `cd` to the project directory:

```
cd project/
```

* For *Event Prediction* model run the following command:
```
python3 check.py --TASK classification --data_folder models/bilevel/results
```

* For *Trend Prediction* model run the following command:
```
python3 check.py --TASK classification --data_folder models/stock_pred/results
```

* For *Simple Trend Prediction* model run the following command:
```
python3 check.py --TASK classification --data_folder models/simple_cls/results
```

* For *Price Prediction* model run the following command:
```
python3 check.py --TASK regression --data_folder models/regression/results --threshold 0.03
```



## Training the model

### 0. Download Data

Pleae download the three datasets from [here](https://drive.google.com/drive/folders/1xKjd9hzA8UTn2DXVIYYnX5TngNAMom19?usp=sharing) and put them in the `data/` folder.

Now, you have three folders with the names of `Domain_adaptation/`, `Event_detection/`, and `Trading_benchmark/` in the `data/` folder. The first two have `train.txt` and `dev.txt` and the third has `evaluate_news.json` file.

### Phase 1: Domain Adaptation

To fine-tune a base BERT model on the stock market data run the following command:

```
python3 run_domain_adapt.py \
    --output_dir=models/bert_bc_adapted \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file=data/Domain_adaptation/train.txt \
    --do_eval \
    --eval_data_file=data/Domain_adaptation/dev.txt \
    --mlm \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 500 \
    --learning_rate 3e-5 \
    --evaluate_during_training \
    --eval_steps 500 \
    --num_train_epochs 20 \
    --max_steps 10000 \
    --logging_first_step
```

### Phase 2: Corporate Event Detection

To further fine-tune the obtained BERT model to predict the corporate events, run the following command:

```
python3 run_event.py \
    --TASK seq \
    --data_dir data/Event_detection \
    --epoch 5 \
    --model_type models/bert_bc_adapted \
    --output_dir models/seq_cls \
    --bert_lr 5e-5  \
    --per_gpu_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512    # for the baseline model "Event Prediction", set this argument to 256
```

### Phase 3: Stock Market Prediction

* To train the *Event Prediction* (the method of [1]) model run the following command:

```
python3 run_event.py \
    --TASK bilevel \
    --data_dir data/Event_detection \
    --epoch 5 \
    --model_type models/bert_bc_adapted \
    --output_dir models/bilevel \
    --bert_lr 5e-5  \
    --per_gpu_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 256 
```

* To train the *Trend Prediction* model run the following command:

```
python3 run_event.py \
    --TASK bi-level_classification \
    --data_dir data/Trading_benchmark/evaluate_news.json \
    --epoch 5 \
    --model_type models/seq_cls \
    --output_dir models/stock_pred \
    --bert_lr 5e-5  \
    --per_gpu_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 
```

* To train the *Simple Trend Prediction* model run the following command:

```
python3 run_event.py \
    --TASK classification \
    --data_dir data/Trading_benchmark/evaluate_news.json \
    --epoch 5 \
    --model_type models/seq_cls \
    --output_dir models/simple_cls \
    --bert_lr 5e-5  \
    --per_gpu_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 
```

* To train the *Price Prediction* model run the following command:

```
python3 run_event.py \
    --TASK regression \
    --data_dir data/Trading_benchmark/evaluate_news.json \
    --epoch 5 \
    --model_type models/seq_cls \
    --output_dir models/regression \
    --bert_lr 5e-5  \
    --per_gpu_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 
```


**Note:** you can run only the evaluation without training on the models by passing the argument `--do_predict` to the previous commands. This evaluation shows the *RPS* values in addition to the *Accuracy* values.





## Reference

[1] Zhou, Zhihan, Liqian Ma, and Han Liu. "Trade the event: Corporate events detection for news-based event-driven trading." arXiv preprint arXiv:2105.12825 (2021).

[2] https://github.com/Zhihan1996/TradeTheEvent