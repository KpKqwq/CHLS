from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import MT5ForConditionalGeneration, MT5Tokenizer,MT5Config
import torch
T5_PATH = 't5-base' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# Input text
text = "To be <extra_id_0> not to be, that is a <extra_id_1>. We need more <extra_id_2>."

encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

# Generaing 20 sequences with maximum length set to 5
outputs = t5_mlm.generate(input_ids=input_ids, 
                          num_beams=20, num_return_sequences=10,
                          )
#print("the raw outputs are",outputs)
#print("the raw token outpouts are",t5_tokenizer.batch_decode(outputs))
#_0_index = text.index('<extra_id_0>')
#_result_prefix = text[:_0_index]
#_result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>
for line in t5_tokenizer.batch_decode(outputs):
    print(line)
print("%"*10)
_end_token_index_prev=0
def _filter(output, end_tokens=['<extra_id_1>','<extra_id_2>',"<extra_id_3>"]):
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    word_hps=[]
    first_index=_txt.index("<extra_id_0>")
    for  i,end_token in enumerate(end_tokens):
        if end_token in _txt:
            second_index=_txt.index(end_token)
        elif "</s>" in _txt and _txt.index("</s>") > first_index:
            second_index=_txt.index("</s>")
        else:
            second_index=-1
        word_hps.append(_txt[first_index+12:second_index].strip())
        first_index=second_index
        if first_index==-1:
            break
    #print(word_hps)
    global text
    result=text.replace("<extra_id_0>",word_hps[0])
    for  i,end_token in enumerate(end_tokens[:-1]):
        if i+1==len(word_hps):
            break
        result=result.replace(end_token,word_hps[i+1])
    return result
results = list(map(_filter, outputs))
for i in results:
    print(i)