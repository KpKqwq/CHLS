#from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import MT5ForConditionalGeneration, MT5Tokenizer,MT5Config
import torch
T5_PATH = 'google/mt5-small' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

t5_tokenizer = MT5Tokenizer.from_pretrained(T5_PATH)
t5_config = MT5Config.from_pretrained(T5_PATH)
#setattr(t5_config,"forced_eos_token_id",250098)
t5_mlm = MT5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)
# Input text
text = "这里风景 <extra_id_0>，山清水秀。"

encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

# Generaing 20 sequences with maximum length set to 5
outputs = t5_mlm.generate(input_ids=input_ids, 
                          num_beams=20, num_return_sequences=10,max_length=60,eos_token_id=250098
                          )
#print(outputs)
#print("the raw outputs are",outputs)
#print("the raw token outpouts are",t5_tokenizer.batch_decode(outputs))
for i in outputs:
    print(i)
for i in t5_tokenizer.batch_decode(outputs):
    print(i)
_0_index = text.index('<extra_id_0>')
_result_prefix = text[:_0_index]
_result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>

def _filter(output, end_token='<extra_id_1>'):
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if end_token in _txt:
        _end_token_index = _txt.index(end_token)
        return _result_prefix + _txt[:_end_token_index] + _result_suffix
    else:
        return _result_prefix + _txt + _result_suffix

results = list(map(_filter, outputs))
for i in results:
    pass
    print(i)