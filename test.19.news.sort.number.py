
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from transformers import AutoTokenizer,AutoModelForMaskedLM
from bert_score import BERTScorer
import numpy as np
import requests
import json
from torch import nn
import torch
from cal_best_oot import real_cal_best_oot,write_all_results
from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("bart-base-chinese",forced_bos_token_id=tokenizer.cls_token_id).cuda().eval()
#good_dict=[line.strip().split("\t")[0].strip() for line in open("../../modern_chinese_word_freq.txt",encoding="utf-8")]
good_dict=[line.strip().split(" ")[0].strip() for line in open("chinese_vocab.txt",encoding="utf-8")]

class ParaBART:
    def __init__(self, device='cuda', min_length=2, max_length=256, checkpoint='bart-base-chinese',model=None,tokenizer=None):
        # Set up model
        self.device = device
        self.min_length = min_length
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model


        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)


    def score(self, srcs, tgts, batch_size=1):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=256,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=256,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list
            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list                   


#all dict
bart_results=open("results/news/sort.number/bart.ahead/news.out.embed.0.1.oot").readlines()
bert_results=open("results/news/sort.number/bert/news.out.embed.0.1.oot").readlines()
dict_results=open("results/news/sort.number/dict/news.out.embed.0.1.oot").readlines()
embed_results=open("results/news/sort.number/embed/news.out.embed.0.1.oot").readlines()



f6_1=open("results/news/sort.number/news.out.embed.0.1_p1.txt","w+")
f6_2=open("results/news/sort.number/news.out.embed.0.1_p3.txt","w+")
f6_3=open("results/news/sort.number/news.out.embed.0.1_probabilites.txt","w+")
f6_4=open("results/news/sort.number/news.out.embed.0.1.best","w+")
f6_5=open("results/news/sort.number/news.out.embed.0.1.oot","w+")



import time
from tqdm import tqdm
for i in tqdm(range(len(bart_results))):

    bart_tmp_string=bart_results[i]
    bert_tmp_string=bert_results[i]
    dict_tmp_string=dict_results[i]
    embed_tmp_string=embed_results[i]

    store_info=bart_tmp_string.split(":::")[0].strip()
    source_word=store_info.split(".")[0].strip()
    oot_str=store_info+" ::: "
    oot_rs=""

    best_str=store_info+" :: "
    best_rs=""

    p1_str="RESULT\t"+store_info+"\t"
    p1_rs=""

    pro_str="RESULT\t"+store_info+"\t"
    pro_rs=""


    bart_words=bart_tmp_string.split(":::")[1].strip().split(";")[:50]
    bert_words=bert_tmp_string.split(":::")[1].strip().split(";")[:50]
    dict_words=dict_tmp_string.split(":::")[1].strip().split(";")[:50]
    embed_words=embed_tmp_string.split(":::")[1].strip().split(";")[:50]



    bart_words_max10=bart_words[:10]
    bert_words_max10=bert_words[:10]
    dict_words_max10=dict_words[:10]
    embed_words_max10=embed_words[:10]


    cands=list(set(bart_words_max10).union(set(bert_words_max10).union(set(dict_words_max10).union(set(embed_words_max10)))))
    all_indexs=[]
    count=0
    for cand in cands:
        if cand==source_word:
            continue
        bart_index=max(len(bart_words),len(bert_words),len(dict_words),len(embed_words)) if cand not in bart_words else bart_words.index(cand)
        bert_index=max(len(bart_words),len(bert_words),len(dict_words),len(embed_words)) if cand not in bert_words else bert_words.index(cand)
        dict_index=max(len(bart_words),len(bert_words),len(dict_words),len(embed_words)) if cand not in dict_words else dict_words.index(cand)
        embed_index=max(len(bart_words),len(bert_words),len(dict_words),len(embed_words)) if cand not in embed_words else embed_words.index(cand)

        all_index=(bart_index+bert_index+dict_index+embed_index)/4

        if i<1:
            print("cand",cand,"source word",source_word)
            print("bart_index",bart_index,"bert_index",bert_index,"dict_index",dict_index,"embed_index",embed_index)
        if ((cand in bart_words) + (cand in bert_words) + (cand in dict_words) + (cand in embed_words))>=1:
            count+=1

        all_indexs.append((cand,all_index))
    
    if i<1:
        sorted_all_indexs_words_tmp=[(word1,index1) for word1,index1 in sorted(all_indexs,key=lambda x: x[1],reverse=False)]
        print(sorted_all_indexs_words_tmp)
        
    sorted_all_indexs_words=[(word1,index1) for word1,index1 in sorted(all_indexs,key=lambda x: x[1],reverse=False)[:10]]

    if (len(sorted_all_indexs_words)==0):
        sorted_all_indexs_words=[
                ("铜铁1",0),
                ("铜铁2",1),
                ("铜铁3",2),
                ("铜铁4",3),
                ("铜铁5",4),
                ("铜铁6",5),
                ("铜铁7",6),
                ("铜铁8",7),
                ("铜铁9",8),
                ("铜铁10",9)
        ]
    p1_rs=sorted_all_indexs_words[0][0]
    #p1_str+="\t"
    p1_str+=p1_rs

    pro_rs="\t".join([word+" "+str(-score1) for word,score1 in sorted_all_indexs_words])
    #pro_str+="\t"
    pro_str+=pro_rs

    best_rs=sorted_all_indexs_words[0][0]
    #best_str+=" "
    best_str+=best_rs

    oot_rs=";".join([word for word,_ in sorted_all_indexs_words])
    #oot_str+=" "
    oot_str+=oot_rs
    f6_1.write(p1_str.strip()+"\n")
    f6_3.write(pro_str.strip()+"\n")
    f6_4.write(best_str.strip()+"\n")
    f6_5.write(oot_str.strip()+"\n")

f6_1.close()
f6_2.close()
f6_3.close()
f6_4.close()
f6_5.close()

from reader import Reader_lexical
from metrics.evaluation import evaluation



work_dir="results/news/sort.number/"
output_SR_file="news.out"
output_score_file="news.out.scores"
embed_quto="0.1"

evaluation_metric = evaluation()
test_golden_file="dataset/Chinese_LS/news_test_gold.txt"


real_cal_best_oot(work_dir,output_SR_file,output_score_file,embed_quto,evaluation_metric,test_golden_file)
