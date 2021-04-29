import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
  return tokenized_sentences

def preprocessing_dataset_new(dataset, label_type):
  label = []
  print(dataset)
  print(label_type)
  for i in dataset[3]: # 정제된 데이터를 활용할 때
    if i == 'blind':
      label.append(100)
    else:
      # label.append(label_type[i])
      label.append(i)
  out_dataset = pd.DataFrame({'sentence':dataset[0],'entity_01':dataset[1],'entity_02':dataset[2],'label':label,})
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data_new(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset_new(dataset, label_type)
  
  return dataset

def tokenized_dataset_new(dataset, tokenizer):
  concat_entity = list(np.array(dataset['sentence'].tolist()))
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      # max_length=100,
      max_length=150,
      add_special_tokens=True,
      )
  print(type(tokenized_sentences))
  return tokenized_sentences

# import pickle as pickle
# import os
# import pandas as pd
# import torch
# from tqdm.auto import tqdm
# from pororo import Pororo

# import numpy as np

# # Dataset 구성.
# class RE_Dataset(torch.utils.data.Dataset):
#   def __init__(self, tokenized_dataset, labels):
#     self.tokenized_dataset = tokenized_dataset
#     self.labels = labels

#   def __getitem__(self, idx):
#     item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
#     item['labels'] = torch.tensor(self.labels[idx])
#     return item

#   def __len__(self):
#     return len(self.labels)

# # 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# # 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.

# # def preprocessing_dataset(dataset, label_type):
# #   label = []
# #   print(dataset[3].values)
# #   print(label_type)
# #   label_type_new=dict([(value, key) for key, value in label_type.items()])
# #   # for i in dataset[8]:
# #   for i in dataset[3]: # 정제된 데이터를 활용할 때
# #     # print('label : ',i)
# #     if i == 'blind':
# #       label.append(100)
# #     else:
# #       # label.append(label_type[i])
# #       # label.append(label_type_new[i])
# #       label.append(i)
# #   # out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
# #   # out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2], 'e1s':dataset[3],'e1e':dataset[4],
# #   #                             'entity_02':dataset[5], 'e2s':dataset[6],'e2e':dataset[7],'label':label})
# #   out_dataset = pd.DataFrame({'sentence':dataset[0],'entity_01':dataset[1],'entity_02':dataset[2],'label':label,})
# #   return out_dataset

# def preprocessing_dataset(dataset, label_type):
#   label = []
#   for i in dataset[8]:
#     if i == 'blind':
#       label.append(100)
#     else:
#       label.append(label_type[i])
#   out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
#   return out_dataset

# # tsv 파일을 불러옵니다.
# def load_data(dataset_dir):
#   # load label_type, classes
#   with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
#     label_type = pickle.load(f)
#   # load dataset
#   dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
#   # preprecessing dataset
#   dataset = preprocessing_dataset(dataset, label_type)
  
#   return dataset

# # bert input을 위한 tokenizing.
# # tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# # baseline code에서는 2가지 부분을 활용했습니다.
# def tokenized_dataset(dataset, tokenizer):
#   concat_entity = []
#   for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
#     temp = ''
#     temp = e01 + '[SEP]' + e02
#     concat_entity.append(temp)
#   tokenized_sentences = tokenizer(
#       concat_entity,
#       list(dataset['sentence']),
#       return_tensors="pt",
#       padding=True,
#       truncation=True,
#       # max_length=100,
#       max_length=150,
#       add_special_tokens=True,
#       )
#   return tokenized_sentences

# def convert_sentence_to_features(train_dataset, tokenizer, max_len):
    
#     max_seq_len=max_len
#     cls_token=tokenizer.cls_token
#     #cls_token_segment_id=tokenizer.cls_token_id
#     cls_token_segment_id=0
#     sep_token=tokenizer.sep_token
#     pad_token=0
#     pad_token_segment_id=tokenizer.pad_token_id
#     sequence_a_segment_id=0
#     add_sep_token=False
#     mask_padding_with_zero=True
    
#     all_input_ids = []
#     all_attention_mask = []
#     all_token_type_ids = []
#     all_e1_mask=[]
#     all_e2_mask=[]

    

#     # for idx in tqdm(range(len(train_dataset))):
#     for idx in tqdm(range(len(train_dataset))):
        
        
#         if train_dataset['e1s'][idx] > train_dataset['e2s'][idx]:
#             train_dataset['sentence'][idx] = train_dataset['sentence'][idx][:train_dataset['e2s'][idx]] + ' <e2> ' + train_dataset['sentence'][idx][train_dataset['e2s'][idx]:train_dataset['e2e'][idx]+1] + ' </e2> ' + train_dataset['sentence'][idx][train_dataset['e2e'][idx]+1:train_dataset['e1s'][idx]] + ' <e1> ' + train_dataset['sentence'][idx][train_dataset['e1s'][idx]:train_dataset['e1e'][idx]+1] + ' </e1> ' + train_dataset['sentence'][idx][train_dataset['e1e'][idx]+1:]
#         else:
#             train_dataset['sentence'][idx] = train_dataset['sentence'][idx][:train_dataset['e1s'][idx]] + ' <e1> ' + train_dataset['sentence'][idx][train_dataset['e1s'][idx]:train_dataset['e1e'][idx]+1] + ' </e1> ' + train_dataset['sentence'][idx][train_dataset['e1e'][idx]+1:train_dataset['e2s'][idx]] + ' <e2> ' + train_dataset['sentence'][idx][train_dataset['e2s'][idx]:train_dataset['e2e'][idx]+1] + ' </e2> ' + train_dataset['sentence'][idx][train_dataset['e2e'][idx]+1:]    

        
#         token = tokenizer.tokenize(train_dataset['sentence'][idx])
        
#         e11_p = token.index("<e1>")  # the start position of entity1
#         e12_p = token.index("</e1>")  # the end position of entity1
#         e21_p = token.index("<e2>")  # the start position of entity2
#         e22_p = token.index("</e2>")  # the end position of entity2

#         token[e11_p] = "$"
#         token[e12_p] = "$"
#         token[e21_p] = "#"
#         token[e22_p] = "#"

#         #print(token)

#         e11_p += 1
#         e12_p += 1
#         e21_p += 1
#         e22_p += 1

#         special_tokens_count = 1

#         if len(token) > max_seq_len - special_tokens_count:
#             token = token[: (max_seq_len - special_tokens_count)]

#         if add_sep_token:
#             token += [sep_token]

#         token_type_ids = [sequence_a_segment_id] * len(token)

#         token = [cls_token] + token
#         token_type_ids = [cls_token_segment_id] + token_type_ids

#         input_ids = tokenizer.convert_tokens_to_ids(token)

#         attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

#         padding_length = max_seq_len - len(input_ids)
#         input_ids = input_ids + ([pad_token] * padding_length)
#         attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
#         token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

#         e1_mask = [0] * len(attention_mask)
#         e2_mask = [0] * len(attention_mask)

#         for i in range(e11_p, e12_p + 1):
#             e1_mask[i] = 1
#         for i in range(e21_p, e22_p + 1):
#             e2_mask[i] = 1

#         assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
#         assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
#             len(attention_mask), max_seq_len
#         )
#         assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
#             len(token_type_ids), max_seq_len
#         )

#         all_input_ids.append(input_ids)
#         all_attention_mask.append(attention_mask)
#         all_token_type_ids.append(token_type_ids)
#         all_e1_mask.append(e1_mask)
#         all_e2_mask.append(e2_mask)
    
#     all_features = {
#         'input_ids' : torch.tensor(all_input_ids),
#         'attention_mask' : torch.tensor(all_attention_mask),
#         'token_type_ids' : torch.tensor(all_token_type_ids),
#         'e1_mask' : torch.tensor(all_e1_mask),
#         'e2_mask' : torch.tensor(all_e2_mask)
#     }
#     train_label = train_dataset['label'].values   
#     return RE_Dataset(all_features, train_label)

# def tokenized_dataset_new(dataset, tokenizer):
#   concat_entity = []
#   for _,sent,e01,e02,s1,e1,s2,e2 in zip(tqdm(range(len(dataset))),dataset['sentence'],dataset['entity_01'], dataset['entity_02'], dataset['e1s'], dataset['e1e'], dataset['e2s'],dataset['e2e']):

#     ner = Pororo(task='ner', lang='ko')
#     ner_01 = ' \ '+ner(e01)[0][1].lower()+' \ '
#     ner_02 = ' ∧ '+ner(e02)[0][1].lower()+' * '

#     entity_01_start, entity_01_end = int(s1),int(e1)
#     entity_02_start, entity_02_end = int(s2), int(e2)

#     if entity_01_start<entity_02_start:
#       sent=sent[:entity_01_start]+'#'+ner_01+sent[entity_01_start:entity_01_end+1]+' # '+sent[entity_01_end+1:entity_02_start]+\
#       '@'+ner_02+sent[entity_02_start:entity_02_end+1]+' @ '+ner_01+sent[entity_02_end+1:]
#     else:
#       sent=sent[:entity_01_start]+'@'+ner_01+sent[entity_01_start:entity_01_end+1]+' @ '+sent[entity_01_end+1:entity_02_start]+\
#       '#'+ner_02+sent[entity_02_start:entity_02_end+1]+' # '+ner_01+sent[entity_02_end+1:]

#     concat_entity.append(sent)

#   tokenized_sentences = tokenizer(
#       concat_entity,
#       list(dataset['sentence']),
#       return_tensors="pt",
#       padding=True,
#       truncation=True,
#       # max_length=100,
#       max_length=350,
#       add_special_tokens=True,
#       )
#   return tokenized_sentences

# def tokenized_dataset_new01(dataset, tokenizer):
#   concat_entity = list(np.array(dataset['sentence'].tolist()))
#   tokenized_sentences = tokenizer(
#       concat_entity,
#       list(dataset['sentence']),
#       return_tensors="pt",
#       padding=True,
#       truncation=True,
#       # max_length=100,
#       max_length=350,
#       add_special_tokens=True,
#       )
#   print(type(tokenized_sentences))
#   return tokenized_sentences