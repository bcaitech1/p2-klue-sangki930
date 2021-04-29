import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertModel, AutoConfig, AutoModelForSequenceClassification
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup as gcw, EarlyStoppingCallback,get_linear_schedule_with_warmup
from load_data import *
from adamp import AdamP

from pathlib import Path
import argparse
from importlib import import_module
import glob
import re
import numpy as np
import random

# from kobert_transformers import get_kobert_model, get_distilkobert_model

import wandb

# from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
# --report-to wandb

# from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import ExponentialLR

from torch.cuda import amp

from customtrainer import *

hyperparameter_defaults = dict(
    dropout = 0.1,
    batch_size = 100,
    learning_rate = 5.62e-5,
    epochs = 1,
    model_name = 'BertForSequenceClassification',
    tokenizer_name = 'BertTokenizer',
    smoothing = 0.2
    )

wandb.init(config=hyperparameter_defaults, project="sweep-test")
config = wandb.config

def increment_output_dir(output_path,exist_ok=False):
  path=Path(output_path)
  if(path.exists() and exist_ok) or (not path.exists()):
    return str(path)
  else:
    dirs = glob.glob(f"{path}")
    matches = [re.search(rf"%s(\d+)" %path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 2
    return f"{path}{n}"

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  wandb.log({'accuracy': acc})
  print('정확도',acc)
  return {
      'accuracy': acc,
  }

def train(args):

  seed_everything(args.seed)

  # wandb.init(project="monologg_kobert")

  # load model and tokenizer
  # MODEL_NAME = "bert-base-multilingual-cased"
  MODEL_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

  # tokenizer = PreTrainedTokenizerFast.from_pretrained("taeminlee/kogpt2")

  # load dataset
  f = '/opt/ml/input/data/train/ner_train_ver2.tsv'
  f2 = '/opt/ml/input/data/test/ner_test_ver2.tsv'
  train_dataset = load_data_new(f)
#   dev_dataset = load_data_new(f2)
  print(train_dataset.shape)
  train_dataset,dev_dataset = train_test_split(train_dataset,test_size=0.2,shuffle=True)
  print(train_dataset.shape)
  # train_dataset = load_data("/opt/ml/input/data/train/train_3.tsv")
  # dev_dataset = load_data("./dataset/train/dev.tsv")
  train_label = train_dataset['label'].values
  print(train_dataset['label'])
  print(dev_dataset['label'])
  dev_label = dev_dataset['label'].values

  print('데이터 셋의 문자열 최대 길이',train_dataset['sentence'].map(lambda x:len(x)).max())
  print('테스트 셋의 문자열 최대 길이',dev_dataset['sentence'].map(lambda x:len(x)).max())

#   print(train_label)
  
  # tokenizing dataset
  # ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
  # tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
  # tokenizer.add_special_tokens({"additional_special_tokens": ["#", "@", '₩', '∧']})
  tokenizer.add_special_tokens({"additional_special_tokens": ["#", "@", 'α', 'β']})


  # make dataset for pytorch.

  # normal dataset

  tokenized_train = tokenized_dataset_new(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset_new(dev_dataset, tokenizer)
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 42
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,config=model_config)
  model.to(device)

  output_dir = increment_output_dir(args.output_dir)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_steps=args.save_steps,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs default=4
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=300,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    # label_smoothing_factor=config.smoothing,
    label_smoothing_factor = 0.5,
    eval_steps = 100,            # evaluation step.
    fp16_backend='amp',
    fp16=True,
    fp16_opt_level ='O1',
    dataloader_num_workers=4,
    # report_to = 'wandb',
    
  )
  optimizer=AdamP(model.parameters(),lr=args.lr,weight_decay=0.01)
  # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
  print('파라미터 확인')
  # print(training_args.fp16,training_args.deepspeed,fp16_backend)
  print("여기까지 오니?")
  

  trainer = MultilabelTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
  )

  # print(trainer.use_amp,trainer.use_apex,trainer.fp16_backend)
#   trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=RE_train_dataset,         # training dataset
#     eval_dataset=RE_dev_dataset,             # evaluation dataset
#     compute_metrics=compute_metrics,         # define metrics function
#   )

  # train model
  trainer.train()
  trainer.evaluate()
  torch.cuda.empty_cache()
  del model
  # trainer.save_model('./results')
  # trainer.save_state()
# seed 고정 
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):
  
  train(args)

if __name__ == '__main__':

  

  torch.cuda.empty_cache()
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', type=str, default='Bert')
  parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased')
  
  parser.add_argument('--epochs', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=5.62e-5)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--warmup_steps', type=int, default=300)               # number of warmup steps for learning rate scheduler
  parser.add_argument('--output_dir', type=str, default='./results/expr')
  parser.add_argument('--save_steps', type=int, default=100)
  parser.add_argument('--save_total_limit', type=int, default=3)
  parser.add_argument('--logging_steps', type=int, default=100)
  parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs

  parser.add_argument('--seed' , type=int , default = 2021)
  parser.add_argument('--learning_rate', type=float, default=5.62e-5)
  parser.add_argument('--dropout', type=float, default=0.1)

  parser.add_argument('--tokenizer_name', type=str, default=0.1)
  parser.add_argument('--model_name',type=str,default='BertForSequenceClassfication')
  parser.add_argument('--smoothing',type=float,default=0.2)

  args = parser.parse_args()
  
  main(args)