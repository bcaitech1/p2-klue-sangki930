# Pstage_03_KLUE_Relation_extraction

### training 모델을 학습시키는 파이썬 파일(토크나이징 X)
* python train.py

### inference
* python inference.py --model_dir=[model_path]
* ex) python inference.py --model_dir=./results/checkpoint-500

### evaluation
* python eval_acc.py

## customtrainer.py : hugging face의 Trainer를 상속하고, cross+focal loss를 구현한 부분
## load_data.py : 전처리, 데이터셋을 구성하는 모듈
## train01.py : pororo라이브러리를 활용한 ner토큰이 있는 데이터를 읽어 학습하는 모듈
## train02.py : StratfiedKFold를 앙상블하여 학습하는 모듈
## hardvoting.ipynb : submission csv로 앙상블하는 파이썬 노트북
