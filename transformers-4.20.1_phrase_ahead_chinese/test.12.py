from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("bart-base-chinese",forced_bos_token_id=tokenizer.cls_token_id).cuda()
import re
from tqdm import tqdm
import torch
refs=[]
hyps=[]
ori_sentences=open("dataset/chinese_ls/annotation_data.csv").readlines()
for line in tqdm(open("dataset/chinese_ls/annotation_data.csv").readlines()):
    text=line.strip().split("\t")[0].strip()
    complex1=line.strip().split("\t")[1].strip()
#    text="".join(tokenizer.tokenize(text))
    text=text.replace(" ","")
    text=text.replace("“",'"')
    text=text.replace("”",'"')
#    print(complex1)
#    print(text)
    #text1="".join(tokenizer.tokenize(text))
    #text1=text1.replace("[UNK]","")
    
    complex1_suffix=text.split(complex1)[1].strip()[:2]
    complex1_prefix=text.split(complex1)[0].strip()
    #complex1_pref=tokenizer.tokenize(text)
    
    pattern=complex1_prefix+".*"+complex1_suffix
    refs.append(line.strip().split("\t")[4].split())
    #text="我非常喜欢你。"
    #complex1="喜欢"
    mask_complex1=text.replace(complex1,"[MASK]")
    prefix_complex1=mask_complex1.split("[MASK]")[0].strip()
    input_ids = tokenizer.encode(mask_complex1, return_tensors='pt')
    input_ids_no_mask=tokenizer.encode(text, return_tensors='pt')
    #prefix_ids = [tokenizer.encode(prefix_complex1, return_tensors='pt')[0][:-1]]
    mask_index = [(input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]]
    prefix_ids=input_ids[0,1:mask_index[0]]
    complex1_ids=tokenizer.encode(complex1)[1:-1]
    attn_len=len(complex1_ids)+len(prefix_ids)+1
#    print(input_ids)
#    print(prefix_ids)
    with torch.no_grad():
        #pred_ids= model.generate(input_ids.cuda(), num_beams=17, max_length=mask_index[0]+4+,num_return_sequences=17,prefix_ids=prefix_ids)
        pred_ids= model.generate(input_ids.cuda(), num_beams=34, max_length=512,num_return_sequences=17,prefix_ids=prefix_ids)
    #pred_ids= model.generate(input_ids_no_mask.cuda(), num_beams=20, max_length=mask_index[0]+4+2+1000,num_return_sequences=10,prefix_ids=prefix_ids,attn_len=attn_len)
    #for i in tokenizer.batch_decode(pred_ids):
    #    print(i)

    #for i in tokenizer.batch_decode(pred_ids,skip_special_tokens=True):
    #    print(i)

    sentences=tokenizer.batch_decode(pred_ids,skip_special_tokens=True)
    hyp=[]
    for sentence in sentences:
        sentence1=sentence.replace(" ","")
        search_result=re.search(pattern,sentence1)
        if search_result==None:
            continue
        search_result=search_result[0].replace(complex1_prefix,"")
        search_result=search_result.replace(complex1_suffix,"")
        hyp.append(search_result)
    hyps.append(hyp)

rec=None
prec=None
tp_sum=0
rec_total=0
prec_total=0
for i in range(len(refs)):
    rec_total+=len(set(refs[i]))
    prec_total+=len(set(hyps[i]))
    tp_sum+=len(set(refs[i])&set(hyps[i]))
print("precision score:",tp_sum/prec_total)
print("recall score",tp_sum/rec_total)

f1=open("results.txt","w+")
for i in range(len(refs)):
    #f1.write(" ".join(refs[i])+"|||"+" ".join(hyps[i])+ori_sentences[i].strip().split("\t")[0]+"\n")
    f1.write(" ".join(refs[i])+"|||"+" ".join(hyps[i])+"|||"+" ".join(list(set(refs[i])&set(hyps[i])))+"\n")
f1.write("recall_score: "+str(tp_sum/rec_total)+"\n")
f1.write("precision score: "+str(tp_sum/prec_total)+"\n")

    
    
    
    
        
        
    