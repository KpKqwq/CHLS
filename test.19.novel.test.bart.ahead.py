from dataclasses import replace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from transformers import AutoTokenizer,AutoModelForMaskedLM
from bert_score import BERTScorer
import numpy as np
import requests
import json
import math
import tkinter
from torch import nn
from bert_utils import generate_bert
from dict_utils import generate_dict,read_dict
from reader import Reader_lexical
from metrics.evaluation import evaluation
from cal_best_oot import real_cal_best_oot,write_all_results

from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese")

def skip_words(s1):
    s1=s1.replace("[PAD]","")
    s1=s1.replace("[MASK]","-")
    s1=s1.replace("[CLS]","")
    s1=s1.replace("[SEP]","")
    s1=s1.replace("[UNK]","-")
    return s1


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
qutos=[",",".","!","?","\"","'",":","[SEP]","[CLS]","[UNK]","[PAD]","[MASK]"]
special_suffix=["，","的","地","呢","吗","啊","得","了","：" ,"。"]

def filter_list(list1):
    new_list=[]
    for i in range(len(list1)):
        if list1[i] not in new_list and list1[i] in good_dict:
            new_list.append(list1[i])
        
    return new_list

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
    sorted_sims_berts_word=[(two_word,two_word_score) for two_word, two_word_score in sorted(sims_bert_word,key=lambda x: x[1],reverse=True)][:11]
    new_sorted_sims_berts_word=[(two_word,two_word_score) for two_word,two_word_score in sorted_sims_berts_word if two_word!=source_word][:10]

    final_words=[items[0] for items in new_sorted_sims_berts_word]
    final_scores=[items[1] for items in new_sorted_sims_berts_word]
    return final_words,final_scores


def sort_max50(ori_text,source_word,cands,generateion_tool,flag_text):
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


def give_real_scores_ahead(tokenizer,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=None,prefix_str=None,max_ahead=1,flag=1):
    beam_size,max_len=outputs.size()
    scores_with_suffix=scores_with_suffix[:,:max_len]
    scores_with_suffix_masks=scores_with_suffix_masks[:,:max_len]

    first_index=prefix_len+2
    last_index=min(first_index+max_ahead,max_len)

    ahead_parts=outputs[:,1:]
    ahead_parts=ahead_parts.reshape(1,-1)[0].tolist()

    ahead_part_tokens=list(map(lambda x:tokenizer.convert_ids_to_tokens(x),ahead_parts))
    #ahead_part_tokens_masks=list(map(lambda x:not x.startswith("Ġ") and x not in qutos,ahead_part_tokens))
    ahead_part_tokens_masks=list(map(lambda x:x.startswith("##") and x not in qutos,ahead_part_tokens))
    ahead_part_tokens_masks=torch.tensor(ahead_part_tokens_masks)
    ahead_part_tokens_masks=ahead_part_tokens_masks.reshape(beam_size,-1)
    scores_with_suffix[:,:-1][ahead_part_tokens_masks]=-math.inf
    scores_with_suffix[scores_with_suffix_masks]=-math.inf
    for j in range(0,first_index):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)

    for j in range(last_index,max_len):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)   

    flat_scores_with_suffix=scores_with_suffix.reshape(1,-1).squeeze(dim=0)
    sorted_scores,sorted_indices=torch.topk(flat_scores_with_suffix,k=beam_size*max_ahead)
    beam_idx=sorted_indices//max_len
    len_idx=(sorted_indices%max_len)

    if flag!=None:
        hope_len=len(prefix_str.strip())+flag
    else:
        hope_len=-1

    hope_outputs=[]
    hope_outputs_scores=[]
    candis=[]

    for i in range(len(beam_idx)):
        if sorted_scores[i].tolist()<-10000:
            continue

        tmp_str1="".join(tokenizer.convert_ids_to_tokens(outputs[beam_idx[i],:(len_idx[i]+1)])).replace("##","")
        tmp_str1=skip_words(tmp_str1).strip()

        if len(tmp_str1)==hope_len:
            # if tmp_str1.split()[-1]=="property":
            #     print(beam_idx[i])
            # print(tmp_str1.split()[-1])
            hope_outputs.append(outputs[beam_idx[i]])
            #print(tgt_dict.string(outputs[beam_idx[i]]),sorted_scores[i])
            hope_outputs_scores.append(sorted_scores[i].tolist())
            candis.append(tmp_str1[len(prefix_str):])
        elif hope_len==-1:

            hope_outputs.append(outputs[beam_idx[i],:(len_idx[i]+1)])
            hope_outputs_scores.append(sorted_scores[i].tolist())
            candis.append(tmp_str1[len(prefix_str):])


    return hope_outputs,hope_outputs_scores,candis


import time
begin_time=time.time()

work_dir="results/novel/sort.number/bart.ahead/"
output_SR_file="novel.out"
embed_quto="0.1"
output_score_file="novel.out.scores"
test_golden_file="dataset/Chinese_LS/novel_test_gold.txt"
reader = Reader_lexical()
eval_dir="dataset/Chinese_LS/novel_test_processed.txt"
reader.create_feature(eval_dir)
evaluation_metric = evaluation()


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

            complex1_suffix=flag_text.split("[FLAG]")[1].strip()[:5]
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

            prefix_str=prefix_complex1
            prefix_ids=tokenizer.encode(prefix_complex1)[1:-1]
            prefix_len=len(prefix_ids)

            complex1_ids=tokenizer.encode(complex1)[1:-1]
            suffix_ids=tokenizer.encode(complex1_suffix)[1:-1]
            attn_len=len(complex1_ids)+len(prefix_ids)+1

            drop_rate=0
            for i in range(1):
                w=0
                if len(complex1)==2:
                    k=0
                elif len(complex1)==3:
                    k=0
                elif len(complex1)==4:
                    k=0
                elif len(complex1)==1:
                    k=1
                    w=0
                with torch.no_grad():
                    pred_ids_sample=None

                    outputs,scores_with_suffix,scores_with_suffix_masks=model.generate(input_ids_no_mask.cuda(), 
                                            num_beams=100, 
                                            min_length=3,
                                            max_length=attn_len+2+20,
                                            num_return_sequences=50,
                                            prefix_ids=prefix_ids,
                                            suffix_ids=suffix_ids,
                                            max_aheads=5,
                                            tokenizer=tokenizer,
                                            #complex_len=len(complex1_ids),
                                            complex_len=1,
                                            attn_len=attn_len+100,
                                            # return_dict_in_generate=True,
                                            # output_scores=True
                                        )

                    outputs=outputs.cpu()
                    scores_with_suffix=scores_with_suffix.cpu()
                    scores_with_suffix_masks=scores_with_suffix_masks.cpu()

                    outputs,outputs_scores,candis=give_real_scores_ahead(tokenizer,
                                                                outputs,
                                                                scores_with_suffix,
                                                                scores_with_suffix_masks,
                                                                suffix_ids,
                                                                prefix_len=prefix_len,
                                                                prefix_str=prefix_str,
                                                                max_ahead=5,
                                                                flag=None)

                    final_outputs=[]
                    final_outputs_scores=[]
                    final_candis=[]
                    max_count_1=0

                    for i in range(len(candis)):
                        if candis[i] not in final_candis and candis[i]!=complex1 and candis[i] in good_dict and candis[i] \
                            not in qutos and candis[i] not in special_suffix:
                            if len(candis[i])>=5:
                                continue
                            if len(candis[i])<=len(complex1) or len(complex1)==1:
                                if len(candis[i])!=3:
                                    if max_count_1<10:
                                        final_candis.append(candis[i])
                                        final_outputs.append(outputs[i])
                                        final_outputs_scores.append(outputs_scores[i])
                                        if len(candis[i])==1:
                                            max_count_1+=1
                                    else:
                                        if len(candis[i])!=1:
                                            final_candis.append(candis[i])
                                            final_outputs.append(outputs[i])
                                            final_outputs_scores.append(outputs_scores[i]) 
                                elif len(complex1)==3:
                                    final_candis.append(candis[i])
                                    final_outputs.append(outputs[i])
                                    final_outputs_scores.append(outputs_scores[i])  
                                else:
                                    pass 

            outputs=final_outputs
            outputs_scores=final_outputs_scores
            candis=final_candis

            list_final_hypo=filter_list(candis)


            if len(list_final_hypo)>2:
                sorted_hyps,sorted_hyps_scores=sort_max50(text,complex1,list_final_hypo,generateion_tool,flag_text)
            else:
                sorted_hyps,sorted_hyps_scores=list_final_hypo,[0.5] if list_final_hypo!=[] else[]

            write_all_results(main_word, instance, target_pos, work_dir+output_SR_file+".embed."+str(embed_quto),
                            sorted_hyps, sorted_hyps_scores, evaluation_metric)
            
real_cal_best_oot(work_dir,output_SR_file,output_score_file,embed_quto,evaluation_metric,test_golden_file)
            



