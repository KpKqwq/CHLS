from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0).cuda()
tok = BartTokenizer.from_pretrained("facebook/bart-base")
print(tok("<sep>"))
example_english_phrase = "I <mask> you very much?"
batch = tok(example_english_phrase, add_special_tokens=True,return_tensors="pt")
generated_ids = model.generate(batch["input_ids"].cuda(),num_beams=5, num_return_sequences=5,max_length=60)
#assert tok.batch_decode(generated_ids, skip_special_tokens=True) == 
#    "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
#]
#print(generated_ids)
for i in generated_ids:
    print(i)
for i in tok.batch_decode(generated_ids,skip_special_tokens=True):
    print(i)