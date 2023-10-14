from dataclasses import replace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer,AutoModelForMaskedLM
from bert_score import BERTScorer
import numpy as np
import requests
import json
import tkinter
from torch import nn
from bert_utils import generate_bert
from dict_utils import generate_dict,read_dict
from reader import Reader_lexical
from metrics.evaluation import evaluation
reader = Reader_lexical()
eval_dir="dataset/Chinese_LS/news_test_processed.txt"
reader.create_feature(eval_dir)
evaluation_metric = evaluation()
from cal_best_oot import real_cal_best_oot
from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese")

import gensim
word_2_vector_model_dir = 'models/merge_sgns_bigram_char300.txt'
model_word2vector = gensim.models.KeyedVectors.load_word2vec_format(word_2_vector_model_dir, binary=False)

def write_all_results(main_word, instance, target_pos, output_results, substitutes, substitutes_scores,
                      evaluation_metric):
    proposed_words = {}
    for substitute_str, score in zip(substitutes, substitutes_scores):
        substitute_lemma=substitute_str.lower().strip()
        max_score = proposed_words.get(substitute_lemma)
        if max_score is None or score > max_score:
            #if pos_filter(pos_vocab,target_pos,substitute_str,substitute_lemma):
            proposed_words[substitute_lemma] = score

    evaluation_metric.write_results(
        output_results + "_probabilites.txt",
        main_word, instance,
        proposed_words
    )
    evaluation_metric.write_results_p1(
        output_results + "_p1.txt",
        main_word, instance,
        proposed_words
    )

    evaluation_metric.write_results_p1(
        output_results + "_p3.txt",
        main_word, instance,
        proposed_words,limit=3
    )

    evaluation_metric.write_results_lex_oot(
        output_results + ".oot",
        main_word, instance,
        proposed_words, limit=10
    )


    evaluation_metric.write_results_lex_best(
        output_results + ".best",
        main_word, instance,
        proposed_words, limit=1
    )



def generate_embed(difficult_word):
    try:
        sim_words = model_word2vector.most_similar(difficult_word,topn=50)
        sim_words = [item[0] for item in sim_words]
    except:
        sim_words=[]

    return sim_words

model = BartForConditionalGeneration.from_pretrained("bart-base-chinese",forced_bos_token_id=tokenizer.cls_token_id).cuda().eval()
#model = BartForConditionalGeneration.from_pretrained("bart-base-chinese",forced_bos_token_id=tokenizer.cls_token_id).eval()
model_name_bert="bert-base-chinese"    
tokenizer_bert = AutoTokenizer.from_pretrained(model_name_bert)
model_bert = AutoModelForMaskedLM.from_pretrained(model_name_bert).cuda().eval()
#model_bert = AutoModelForMaskedLM.from_pretrained(model_name_bert).eval()
dict1=read_dict("dataset/dict/HIT-dict=.txt")
print("[finish loading the model]")
#model = BartForConditionalGeneration.from_pretrained("bart-base-chinese").cuda().eval()
import re
from tqdm import tqdm
import torch
refs=[]
hyps=[]
good_dict=[line.strip().split(" ")[0].strip() for line in open("chinese_vocab.txt",encoding="utf-8")]
special_suffix=["，","且","和","并","的","地","呢","吗","啊","：" ,"。"]


def filter(set1):
    set_lst=list(set1)
    new_set=set()
    for i in set_lst:
        if i in good_dict:
            new_set.update([i])
    return new_set


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

generateion_tool=ParaBART(model=model,tokenizer=tokenizer)
bert_score=BERTScorer(lang="zh",rescale_with_baseline=True,use_fast_tokenizer=False)

def sorted_generate(ori_text,source_word,cands):
    for word in cands:
        new_text=re.sub(source_word,word,ori_text)
        print(new_text)
def sort_max10(ori_text,source_word,cands,generateion_tool,flag_text):
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
        new_sentence=re.sub("[FLAG]",word,flag_text)
        _,bert_R1,bert_F1 = bert_score.score([flag_text.replace("[FLAG]",word)],[ori_text],verbose=False)
        bert_scores.append(bert_R1.tolist()[0])
        bart_scores.append(np.exp(generateion_tool.score([ori_text],[flag_text.replace("[FLAG]",word)], batch_size=1)[0]))
    sacle_bert_scores=((np.array(bert_scores)-min(bert_scores)))/(max(bert_scores)-min(bert_scores))
    sims_bert_word=[(cands[i],(bert_quto*sacle_bert_scores[i]+bart_quto*bart_scores[i])/2) for i in range(len(cands))]  
    sorted_sims_berts_word=[(two_word,two_word_score) for two_word, two_word_score in sorted(sims_bert_word,key=lambda x: x[1],reverse=True)][:51]
    new_sorted_sims_berts_word=[(two_word,two_word_score) for two_word,two_word_score in sorted_sims_berts_word if two_word!=source_word][:50]

    final_words=[items[0] for items in new_sorted_sims_berts_word]
    final_scores=[items[1] for items in new_sorted_sims_berts_word]


    return final_words,final_scores


import time
begin_time=time.time()

work_dir="results/news/sort.number/embed/"
output_SR_file="news.out"
embed_quto="0.1"
output_score_file="news.out.scores"
test_golden_file="dataset/Chinese_LS/news_test_gold.txt"

count_gen=-1
from tqdm import tqdm
for main_word in tqdm(reader.words_candidate):
    count_gen+=1
    # if count_gen==2:
    #     break
    for instance in reader.words_candidate[main_word]:
        for context in reader.words_candidate[main_word][instance]:
            text = context[1]
            original_text = text
            original_words = text.split(' ')
            index_word = int(context[2])
            target_word = text.split(' ')[index_word]
            target_pos = main_word.split('.')[-1]
            target_lemma=target_word.lower().strip()

            text=text.replace("“",'"')
            text=text.replace("’","'")
            text=text.replace("‘","'")
            text=text.replace("”",'"')
            text_list=original_words
            text_index=index_word
            complex1=target_word

            text=text.replace(" ","")

            text=text.lower()
            complex1=complex1.lower()

            flag_text_list=text_list
            flag_text_list[text_index]="[FLAG]"
            flag_text=" ".join(flag_text_list)
            flag_text=flag_text.replace(" ","")

            complex1_suffix=flag_text.split("[FLAG]")[1].strip()[:1]
            complex1_prefix=flag_text.split("[FLAG]")[0].strip()

            complex1_prefix=tokenizer.decode(tokenizer.encode(complex1_prefix),skip_special_tokens=True).replace(" ","")


            if "?" in complex1_suffix:
                complex1_suffix=re.sub("\?","\?",complex1_suffix)
            pattern=".*?"+complex1_suffix


            mask_complex1_list=text_list
            mask_complex1_list[text_index]="[MASK]"
            mask_complex1=" ".join(mask_complex1_list)
            mask_complex1=mask_complex1.replace(" ","")

            prefix_complex1=flag_text.split("[FLAG]")[0].strip()

            input_ids_no_mask=tokenizer.encode(text, return_tensors='pt')
            input_ids=input_ids_no_mask
            max_ori_len=len(input_ids_no_mask[0])
            prefix_ids = [tokenizer.encode(prefix_complex1, return_tensors='pt')[0][:-1]][0]

            complex1_ids=tokenizer.encode(complex1)[1:-1]
            attn_len=len(complex1_ids)+len(prefix_ids)+1

            final_hyp=set()

            final_hyp_embed=generate_embed(complex1)
            list_final_hypo_embed=list(filter(set(final_hyp_embed)))

            if len(list_final_hypo_embed)>2:
                sorted_hyps,sorted_hyps_scores=sort_max10(text,complex1,list_final_hypo_embed,generateion_tool,flag_text)
            else:
                sorted_hyps,sorted_hyps_scores=list_final_hypo_embed,[0.5] if list_final_hypo_embed!=[] else[]


            write_all_results(main_word, instance, target_pos, work_dir+output_SR_file+".embed."+str(embed_quto),
                            sorted_hyps, sorted_hyps_scores, evaluation_metric)
            
real_cal_best_oot(work_dir,output_SR_file,output_score_file,embed_quto,evaluation_metric,test_golden_file)
            



