from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForMaskedLM.from_pretrained("bert-base-chinese").cuda()
inputs = tokenizer("句子太长，念起来[MASK][MASK]。", return_tensors='pt')
#print(inputs.input_ids)
inputs["input_ids"]=inputs["input_ids"].cuda()
inputs["token_type_ids"]=inputs["token_type_ids"].cuda()
inputs["attention_mask"]=inputs["attention_mask"].cuda()
logits = model(**inputs).logits
final_words=[]
final_results=[]
for i in range(len((inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0])):
    mask_token_index = [(inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][1]]
    predicted_token_id = logits[0, mask_token_index].argsort(axis=-1)[0].cpu().tolist()[::-1][:20]
    final_words.append(tokenizer.batch_decode(predicted_token_id))
def print1(i,words):
    if i==2:
        final_results.append(words)
        return 
    else:
        for word in final_words[i]:
            print1(i+1,words+word)
print1(0,"")    

import pickle
with open("final_dict","rb") as f1:
    final_dict=pickle.load(f1)
for words in final_results:
    if words in final_dict:
        print(words)