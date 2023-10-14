from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("bart-base-chinese",forced_bos_token_id=tokenizer.cls_token_id).cuda()
text="我非常喜欢你。"
complex1="喜欢"
mask_complex1=text.replace(complex1,"[MASK]")
prefix_complex1=mask_complex1.split("[MASK]")[0].strip()
input_ids = tokenizer.encode(mask_complex1, return_tensors='pt')
input_ids_no_mask=tokenizer.encode(text, return_tensors='pt')
#prefix_ids = [tokenizer.encode(prefix_complex1, return_tensors='pt')[0][:-1]]
mask_index = [(input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]]
prefix_ids=input_ids[0,1:mask_index[0]]
complex1_ids=tokenizer.encode(complex1)[1:-1]
attn_len=len(complex1_ids)+len(prefix_ids)+1
print(input_ids)
print(prefix_ids)
pred_ids= model.generate(input_ids.cuda(), num_beams=20, max_length=mask_index[0]+4+2+1000,num_return_sequences=20,prefix_ids=prefix_ids)
#pred_ids= model.generate(input_ids_no_mask.cuda(), num_beams=20, max_length=mask_index[0]+4+2+1000,num_return_sequences=10,prefix_ids=prefix_ids,attn_len=attn_len)
#for i in tokenizer.batch_decode(pred_ids):
#    print(i)

for i in tokenizer.batch_decode(pred_ids,skip_special_tokens=True):
    print(i)