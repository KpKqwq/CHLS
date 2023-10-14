from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("t5-base")
example_english_phrase = "Today is a good day, we <extra_id_0> to play,how about you?"
#example_english_phrase = "<extra_id_0>"
batch = tok(example_english_phrase, return_tensors="pt")
#print(batch)
model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
generated_ids = model.generate(batch["input_ids"].cuda(),num_beams=20, num_return_sequences=10)
#assert tok.batch_decode(generated_ids, skip_special_tokens=True) == 
#    "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
#]
print(generated_ids)
for i in generated_ids:
    print(i)
#print(generated_ids.shape)
#print(tok.decode(generated_ids[0]))