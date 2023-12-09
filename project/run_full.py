from tqdm import tqdm, trange
import logging
import argparse
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

from transformers import BertForSequenceClassification, BertConfig, BertTokenizerFast, AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data import StockDataset, load_and_cache_benchmark_dataset, load_and_cache_dataset, load_and_cache_predict_dataset, NewsDataset
from utils.model import StockBertForSequenceClassification



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(test_dataset, model, args):
    logger.info('Start Evaluating')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_gpu_batch_size * args.n_gpu, sampler=test_sampler)

    # if not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)

    model.eval()
    test_iterator = tqdm(test_dataloader, desc="Iteration")
    correct = 0
    all_model_preds = torch.zeros([0], dtype=torch.uint8)
    all_labels = torch.zeros([0], dtype=torch.uint8)

    with torch.no_grad():
        for batch in test_iterator:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask)
            model_preds = outputs[0].argmax(dim=1)
            equal_mask = model_preds == labels
            correct += torch.count_nonzero(equal_mask)

            all_model_preds = torch.cat([all_model_preds, model_preds.cpu().type_as(all_model_preds)], dim=0)
            all_labels = torch.cat([all_labels, labels.cpu().type_as(all_labels)], dim=0)

    source_path = os.path.join(args.output_dir, 'results') if args.predict_dir == '' else args.predict_dir
    if not os.path.exists(source_path):
        os.makedirs(source_path)

    np.save(os.path.join(source_path, 'model_preds.npy'), all_model_preds)

    logger.info('\n')

    logger.info('Accuracy: {}'.format(100. * correct / len(all_labels)))
    
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels.numpy(), all_model_preds.numpy(), average='micro')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels.numpy(), all_model_preds.numpy(), average='macro')
    weigh_precision, weigh_recall, weigh_f1, _ = precision_recall_fscore_support(all_labels.numpy(), all_model_preds.numpy(), average='weighted')
    
    logger.info(f'Micro\t Precision={100.*micro_precision:.2f} Recall={100.*micro_recall:.2f} F1={100.*micro_f1:.2f}')
    logger.info(f'Macro\t Precision={100.*macro_precision:.2f} Recall={100.*macro_recall:.2f} F1={100.*macro_f1:.2f}')
    logger.info(f'Weighted\t Precision={100.*weigh_precision:.2f} Recall={100.*weigh_recall:.2f} F1={100.*weigh_f1:.2f}')


def main():
    # config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default='data/Trading_benchmark',
        type=str,
        help="The input data dir. Should contain the event detection data",
    )
    parser.add_argument(
        "--model_type",
        default='bert-base-cased',
        type=str,
        help="The model type",
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="Add the argument during backtesting on news"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--predict_dir",
        default='',
        type=str,
        help="The directory to save predict result. Use it with --do_predict",
    )
    parser.add_argument(
        "--max_seq_length", default=512, type=int, help="Max sequence length for prediction"
    )
    parser.add_argument(
        "--bert_lr", default=5e-5, type=float, help="The peak learning rate for BERT."
    )
    parser.add_argument(
        "--epoch", default=5, type=int, help="Number of epoch for training"
    )
    parser.add_argument(
        "--num_labels", default=12, type=int, help="Number of unique labels in the dataset"
    )
    parser.add_argument(
        "--per_gpu_batch_size", default=8, type=int, help="Batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", default=1, type=int, help="Batch size"
    )
    parser.add_argument(
        "--seed", default=24, type=int, help="Random seed"
    )
    parser.add_argument(
        "--n_gpu", default=4, type=int, help="Number of GPUs"
    )
    parser.add_argument(
        "--device", default='cpu', type=str, help="Number of GPUs"
    )

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initialize
    set_seed(args.seed)

    # load data
    logger.info('Processing and loading data')
    cache_path = os.path.join(os.path.dirname(args.data_dir), 'cached_train_test_{}'.format(args.max_seq_length))
    if not os.path.exists(cache_path):
        load_and_cache_benchmark_dataset(args.data_dir, args.model_type, args.max_seq_length, args.seed)

    with open(cache_path, 'rb') as f:
        dataset = pickle.load(f)
        train_dataset = StockDataset(dataset[0], dataset[1])
        val_dataset = StockDataset(dataset[2], dataset[3])

    logger.info(
        'Total training batch size: {}'.format(args.per_gpu_batch_size * args.gradient_accumulation_steps * args.n_gpu))

    # load model
    config = BertConfig.from_pretrained(args.model_type)
    config.num_labels = args.num_labels
    config.max_seq_length = args.max_seq_length
    config.stock_labels = 3
    model = StockBertForSequenceClassification.from_pretrained(args.model_type, config=config)
    model.to(args.device)

    # handle predict
    if args.do_predict:
        evaluate(val_dataset, model, args)
        return

    optim = AdamW(model.parameters(), lr=args.bert_lr)
    total_steps = int(
        len(train_dataset) * args.epoch / (args.per_gpu_batch_size * args.gradient_accumulation_steps * args.n_gpu))
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)

    # training
    logger.info('Start Training')
    logger.info(args)
    logger.info('Total Optimization Step: ' + str(total_steps))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_batch_size * args.n_gpu, sampler=train_sampler,
                                  num_workers=4, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    model.zero_grad()

    epochs_trained = 0
    train_iterator = trange(epochs_trained, args.epoch, desc="Epoch")

    set_seed(args.seed)  # add here for reproducibility

    for epoch in train_iterator:
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            optim.zero_grad()

            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0].mean()
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optim.step()
                scheduler.step()

        # evaluation
        evaluate(val_dataset, model, args)

        # save model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_dir = os.path.join(args.output_dir, f'checkpoint{epoch:02d}')

        logger.info("Saving model checkpoint to %s", save_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(save_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_type)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()