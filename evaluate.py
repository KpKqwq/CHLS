import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("bart-base-chinese",forced_bos_token_id=tokenizer.cls_token_id).cuda().eval()
#model = BartForConditionalGeneration.from_pretrained("bart-base-chinese").cuda().eval()
import re
from tqdm import tqdm
import torch

refs=[]
hyps=[]
good_dict=[line.strip().split("\t")[0].strip() for line in open("modern_chinese_word_freq.txt")]
def reduce_fuc(set1):
    pass
def filter(set1):
    set_lst=list(set1)
    new_set=set()
    for i in set_lst:
        if i in good_dict:
            new_set.update([i])
    return new_set
def find_good_string(s1):
    for quto in ["，","。",'"',"'","！","？"]:
        if quto in s1:
            s1=s1.replace(quto,"")
    tmp_string=""
    for i in range(len(s1)):
        if s1[i] in ["，","。",'"',"'","！","？"]:
            continue
        else:
            tmp_string+=s1[i]
        if i==0:
            continue
        if tmp_string in good_dict:
            if len(tmp_string)!=1:
                return tmp_string
    return ""
def find_good_string_behind(s1):
    for quto in ["，","。",'"',"'","！","？"]:
        if quto in s1:
            s1=s1.replace(quto,"")
    tmp_string=s1
    for i in range(len(s1)):
        if tmp_string in good_dict:
            if len(tmp_string)!=1:
                return tmp_string
        tmp_i=len(s1)-i
        if tmp_i==1:
            break
        tmp_string=s1[:tmp_i]
    return ""

ori_sentences=open("dataset/chinese_ls/annotation_data.csv").readlines()
ori_sentneces_split=ori_sentences[:20]+ori_sentences[168:198]
def evalaute_lexical(model,tokenizer,ori_sentences):
    hyp_sentences=[]
    for line in tqdm(ori_sentences):
        text=line.strip().split("\t")[0].strip()
        complex1=line.strip().split("\t")[1].strip()
        text=text.replace(" ","")
        text=text.replace("“",'"')
        text=text.replace("”",'"')
        text=text.lower()
        
        complex1_suffix=text.split(complex1)[1].strip()[:1]
        complex1_prefix=text.split(complex1)[0].strip()
        
        complex1_prefix=tokenizer.decode(tokenizer.encode(complex1_prefix),skip_special_tokens=True).replace(" ","")
        pattern=".*?"+complex1_suffix
        refs.append(line.strip().split("\t")[4].split())
        mask_complex1=text.replace(complex1,"[MASK]")
        prefix_complex1=mask_complex1.split("[MASK]")[0].strip()
        input_ids = tokenizer.encode(mask_complex1, return_tensors='pt')
        input_ids_no_mask=tokenizer.encode(text, return_tensors='pt')
        mask_index = [(input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]]
        prefix_ids=input_ids[0,1:mask_index[0]]
        complex1_ids=tokenizer.encode(complex1)[1:-1]
        attn_len=len(complex1_ids)+len(prefix_ids)+1
        final_hyp=set()
        drop_rate=0
        for i in range(1):
            if len(complex1)==2:
                k=0
            elif len(complex1)==3:
                k=0
            elif len(complex1)==4:
                k=0
            elif len(complex1)==1:
                k=1
            with torch.no_grad():
                pred_ids_sample=None
                pred_ids= model.generate(input_ids_no_mask.cuda(), num_beams=20, max_length=attn_len+2+k+100,num_return_sequences=10,prefix_ids=prefix_ids,mask_pos=range(len(prefix_ids)+1,attn_len), drop_rate=drop_rate,complex_len=len(complex1_ids),temperature=8.0,attn_len=attn_len+k+100,do_sample=False)
                sentences=tokenizer.batch_decode(pred_ids,skip_special_tokens=True)
                sentences_sample=[]
                if pred_ids_sample!=None:
                    sentences_sample=tokenizer.batch_decode(pred_ids_sample,skip_special_tokens=True)

                sentences+=sentences_sample
                hyp=[]
                hyp_sentence=[]
                for sentence in sentences:
                    sentence1=sentence.replace(" ","")
                    sentence1=sentence1.replace("?","？")
                    sentence1=sentence1.replace("!","！")
                    sentence1=sentence1.replace(",","，")
                    hyp_sentence.append(sentence1)
                    sentence1=sentence1.lstrip(complex1_prefix).strip()
                    search_result=re.search(pattern,sentence1)
                    if search_result==None:
                        if len(complex1)!=4:                     
                            search_result_fake=sentence1
                            search_result_fake=find_good_string(search_result_fake)
                            if search_result_fake!="":
                                hyp.append(search_result_fake)
                            continue
                        else:
                            search_result_fake=sentence1
                            search_result_fake=find_good_string_behind(search_result_fake)
                            if search_result_fake!="":
                                hyp.append(search_result_fake)
                            continue
                    search_result=search_result[0].replace(complex1_suffix,"").strip()                
                    hyp.append(search_result)
                final_hyp=final_hyp.union(set(hyp))
                hyp_sentences.append(hyp_sentence)
        hyps.append(list(filter(final_hyp)))
    rec=None
    prec=None
    tp_sum=0
    rec_total=0
    prec_total=0
    potential=0
    for i in range(len(refs)):
        rec_total+=len(set(refs[i]))
        prec_total+=len(set(hyps[i]))
        tp_sum+=len(set(refs[i])&set(hyps[i]))
        if len(set(refs[i])&set(hyps[i]))>0:
            potential+=1
    print("precision score:",tp_sum/prec_total)
    print("recall score",tp_sum/rec_total)
    pre1=round(tp_sum/prec_total,2)
    rc1=round(tp_sum/rec_total,2)
    print("f1 score",(2*pre1*rc1)/(pre1+rc1))
    print("potential score",potential/524)
    f1=open("AAAevaluate.txt","w+")
    for i in range(len(refs)):
    #f1.write(" ".join(refs[i])+"|||"+" ".join(hyps[i])+ori_sentences[i].strip().split("\t")[0]+"\n")
        f1.write(" ".join(refs[i])+"|||"+" ".join(list(set(hyps[i])))+"|||"+" ".join(list(set(refs[i])&set(hyps[i])))+"\n")


    
evalaute_lexical(model,tokenizer,ori_sentneces_split)
    
        
        
    