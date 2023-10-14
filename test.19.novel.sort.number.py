
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

import re
bert_score=None
#bert_score=BERTScorer(lang="zh",rescale_with_baseline=True,use_fast_tokenizer=False)
def sort_max10(ori_text,source_word,cands,generateion_tool):
    hownet_quto=1
    embed_quto=1
    bert_quto=1
    bart_quto=11
    bert_scores=[]
    sims=[]
    bart_scores=[]
    new_word_list=[]
    hownet_scores=[]
    hownet_scores_ori=[]

    for word in cands:
        new_sentence=re.sub(source_word,word,ori_text)
        _,bert_R1,bert_F1 = bert_score.score([ori_text.replace(source_word,word)],[ori_text],verbose=False)
        bert_scores.append(bert_R1.tolist()[0])
        bart_scores.append(np.exp(generateion_tool.score([ori_text],[ori_text.replace(source_word,word)], batch_size=1)[0]))
    sacle_bert_scores=((np.array(bert_scores)-min(bert_scores)))/(max(bert_scores)-min(bert_scores))
    sims_bert_word=[(cands[i],(bert_quto*sacle_bert_scores[i]+bart_quto*bart_scores[i])/2) for i in range(len(cands))]  
    sorted_sims_berts_word=[two_word for two_word, _ in sorted(sims_bert_word,key=lambda x: x[1],reverse=True)]
    new_sorted_sims_berts_word=[two_word for two_word in sorted_sims_berts_word if two_word!=source_word and two_word in good_dict]
    return new_sorted_sims_berts_word



#all dict
bart_results=open("results/novel/sort.number/bart.ahead/novel.out.embed.0.1.oot").readlines()
bert_results=open("results/novel/sort.number/bert/novel.out.embed.0.1.oot").readlines()
dict_results=open("results/novel/sort.number/dict/novel.out.embed.0.1.oot").readlines()
embed_results=open("results/novel/sort.number/embed/novel.out.embed.0.1.oot").readlines()

#all embed


#all bert

f6_1=open("results/novel/sort.number/novel.out.embed.0.1_p1.txt","w+")
f6_2=open("results/novel/sort.number/novel.out.embed.0.1_p3.txt","w+")
f6_3=open("results/novel/sort.number/novel.out.embed.0.1_probabilites.txt","w+")
f6_4=open("results/novel/sort.number/novel.out.embed.0.1.best","w+")
f6_5=open("results/novel/sort.number/novel.out.embed.0.1.oot","w+")



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
            #print(cand)
            count+=1

        all_indexs.append((cand,all_index))
    
    # if i<5:
    #     print("*"*100)
    #print("the four word length",count)
    if i<1:
        sorted_all_indexs_words_tmp=[(word1,index1) for word1,index1 in sorted(all_indexs,key=lambda x: x[1],reverse=False)]
        print(sorted_all_indexs_words_tmp)
        
    #sorted_all_indexs_words=[word1 for word1,index1 in sorted(all_indexs,key=lambda x: x[1],reverse=False)[:10]]
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

    f6_1.flush()
    f6_3.flush()
    f6_4.flush()
    f6_5.flush()
    

f6_1.close()
f6_2.close()
f6_3.close()
f6_4.close()
f6_5.close()

from reader import Reader_lexical
from metrics.evaluation import evaluation




work_dir="results/novel/sort.number/"
output_SR_file="novel.out"
output_score_file="novel.out.scores"
embed_quto="0.1"



evaluation_metric = evaluation()
test_golden_file="dataset/Chinese_LS/novel_test_gold.txt"


real_cal_best_oot(work_dir,output_SR_file,output_score_file,embed_quto,evaluation_metric,test_golden_file)
