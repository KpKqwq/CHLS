<<<<<<< HEAD
<<<<<<< HEAD
# Chinese Lexical Substitution: Dataset and Method


# install 
## install transformers
Our method is build on transformers, based on custom modification of scripts. You need first follow the commands to install transformers
cd transformers-4.20.1_phrase_ahead_chinese
pip install -e .

## other dependencies

pip install -r requirements.txt

# Models 

## [Embedding Model](https://github.com/Embedding/Chinese-Word-Vectors), placed in models/

## [DICT](https://pan.baidu.com/share/link?shareid=2858555949&uk=2738088569),haved placed in dataset/dict/

## [Paraphraser](https://drive.google.com/file/d/1pXYDbVJQnVzjcLwGJzSWbX0dFtBqRStm/view?usp=sharing), placed in bart-base-chinese/


## [BERT](https://huggingface.co/bert-base-chinese), placed in bert-base-chinese/


# Run example for news category 

## Dict
python test.19.news.test.dictonary.py
## Embedding 
python test.19.news.test.embed.py
## ParaLS
python test.19.news.test.bart.ahead.py
## BERT
python test.19.news.test.bert.py
## Our Emsemble method
python test.19.news.sort.number.py


# Results
![](PGLS.jpg)

# 
=======
# CHLS
>>>>>>> b3be66a3f25823fcd10aadabce5b7b729dbbf132
=======
# CHLS
>>>>>>> 797fc8f9240e746b11b88aaf884b6e0b62c03155
