from transformers import MBartTokenizer, MBartForConditionalGeneration

tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25').cuda()

text = '<s> 我 <mask> 你 . </s> zh_CN'
#inputs = tokenizer.prepare_translation_batch([text], src_lang='en_XX')
inputs=tokenizer(text,add_special_tokens=False,return_tensors="pt")
outputs = model.generate(inputs['input_ids'].cuda(), decoder_start_token_id=tokenizer.lang_code_to_id['zh_CN'],num_return_sequences=2,
                         num_beams=5)
for i in tokenizer.batch_decode(outputs):
    print(i)