from transformers import MBartForConditionalGeneration, BartTokenizer
import torch
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0).cuda()
tok = BartTokenizer.from_pretrained("facebook/bart-base")
print(tok("<sep>"))
example_english_phrase = "I <mask> you"
batch = tok(example_english_phrase, add_special_tokens=True,return_tensors="pt")
input_mask=batch["input_ids"]==tok.mask_token_id
input_ids=batch["input_ids"].cuda()
attention_mask=batch["attention_mask"].cuda()
output=model(input_ids,attention_mask)

mask_logits=output.logits.squeeze()[input_mask.squeeze()].squeeze().cpu()
#
_,indexs=torch.topk(mask_logits,k=5,dim=-1)
print(indexs)
words=[tok.decode([index],skip_special_tokens=True) for index in indexs.squeeze()]
print(words)
#generated_ids = model.generate(batch["input_ids"].cuda(),num_beams=2, num_return_sequences=2,max_length=60)
#assert tok.batch_decode(generated_ids, skip_special_tokens=True) == 
#    "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
#]
#print(generated_ids)
#for i in generated_ids:
#    print(i)
#for i in tok.batch_decode(generated_ids,skip_special_tokens=True):
#    print(i)