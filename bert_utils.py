from ctypes.wintypes import WORD
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer

logger = logging.getLogger(__name__)
import time
begin_time=time.time()
def read_eval_dataset(data_path):
    sentences = []
    mask_words = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            row = line.strip().split('\t')
            sentence, mask_word = row[0], row[1]
            sentences.append(''.join(sentence.split(' ')))
            mask_words.append(mask_word)
    return sentences, mask_words

def read_dict(dict_path):
    dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, value = line.strip().split(',')
            dict[key] = value
    return dict

def encoder(tokenizer, sequence_a, sequence_b, max_length):
    sequence_dict = tokenizer.encode_plus(sequence_a, sequence_b, max_length=max_length, padding=True, return_tensors='pt')
    return sequence_dict

def truncate(sentence, start_index, end_index, window):
    # extract words around the content word
    len_sent = len(sentence)
    len_word = end_index - start_index
    radius = int((window - len_word) / 2)
    word_half_index = int((start_index + end_index) / 2)
    if start_index - radius < 0:
        sentence = sentence[0:window-1]
    elif end_index + radius > len_sent - 1:
        sentence = sentence[len_sent-window-1:len_sent-1]
    else:
        sentence = sentence[start_index-radius:end_index+radius]
    return sentence

def predict_char(tokenizer, model, sentence, mask_sentence, max_length, k,flag_text=None):
    sequence_dict = encoder(tokenizer, sentence, mask_sentence, max_length)
    input_ids = sequence_dict['input_ids'].to('cuda')
    attention_masks = sequence_dict['attention_mask'].to('cuda')
    token_type_ids = sequence_dict['token_type_ids'].to('cuda')
    masked_index = int(torch.where(input_ids == tokenizer.mask_token_id)[1][0])
    with torch.no_grad():
        outputs = model(input_ids, attention_masks, token_type_ids) # Return type: tuple(torch.FloatTensor) comprising various elements depending on the configuration (BertConfig) and inputs
    token_logits = outputs[0]
    mask_token_logits = token_logits[0, masked_index, :]
    mask_token_probs = mask_token_logits.softmax(dim=0)
    top_k_ids = torch.topk(mask_token_logits, k).indices.tolist()
    logits = mask_token_logits[top_k_ids]
    probs = mask_token_probs[top_k_ids]
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids)
    return probs, top_k_tokens


# def predict_char_log(tokenizer, model, sentence, mask_sentence, max_length, k,flag_text=None):
#     sequence_dict = encoder(tokenizer, sentence, mask_sentence, max_length)
#     input_ids = sequence_dict['input_ids'].to('cuda')
#     attention_masks = sequence_dict['attention_mask'].to('cuda')
#     token_type_ids = sequence_dict['token_type_ids'].to('cuda')
#     masked_index = int(torch.where(input_ids == tokenizer.mask_token_id)[1][0])
#     with torch.no_grad():
#         outputs = model(input_ids, attention_masks, token_type_ids) # Return type: tuple(torch.FloatTensor) comprising various elements depending on the configuration (BertConfig) and inputs
#     token_logits = outputs[0]
#     mask_token_logits = token_logits[0, masked_index, :]
#     mask_token_probs = mask_token_logits.softmax(dim=0)
#     top_k_ids = torch.topk(mask_token_logits, k).indices.tolist()
#     logits = mask_token_logits[top_k_ids]
#     probs = mask_token_probs[top_k_ids]
#     top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids)
#     return torch.log(probs), top_k_tokens






def get_idiom_subs(tokenizer, model, source_sent, mask_word, max_length,max_words=None,flag_text=None):
    if flag_text==None:
        mask_sentence = source_sent.replace(mask_word, '[MASK]'*4)
    else:
        mask_sentence = flag_text.replace("[FLAG]", '[MASK]'*4)
    probs, top_5_tokens = predict_char(tokenizer, model, source_sent, mask_sentence, max_length, max_words)
    for _ in range(3):
        for i in range(max_words):
            temp_sentence = mask_sentence.replace('[MASK]', top_5_tokens[i], 1)
            probs_sub, top_token = predict_char(tokenizer, model, source_sent, temp_sentence, max_length, 1)
            probs[i] *= probs_sub[0]
            top_5_tokens[i] += top_token[0]
    return probs, top_5_tokens











def get_word_subs(tokenizer, model, source_sent, mask_word, max_length,max_words=None,flag_text=None):
    if flag_text==None:
        mask_sentence = source_sent.replace(mask_word, '[MASK]'*2)
    else:
        mask_sentence = flag_text.replace("[FLAG]", '[MASK]'*2)
    probs_first, top_5_tokens = predict_char(tokenizer, model, source_sent, mask_sentence, max_length, max_words)
    substitution_words = []
    word_probs = []
    for i in range(max_words):
        temp_sentence = mask_sentence.replace('[MASK]', top_5_tokens[i], 1)
        probs_second, top_3_tokens = predict_char(tokenizer, model, source_sent, temp_sentence, max_length, 3)
        #probs_second, top_3_tokens = predict_char(tokenizer, model, source_sent, temp_sentence, max_length, 2)
        probs = [(probs_first[i] * next_prob).item() for next_prob in probs_second]
        words = [top_5_tokens[i] + next_char for next_char in top_3_tokens]
        word_probs += probs
        substitution_words += words

    return word_probs, substitution_words

def get_char_subs(tokenizer, model, source_sent, mask_char, max_length, k,flag_text=None):
    if flag_text==None:
        mask_sentence = source_sent.replace(mask_char, '[MASK]')
    else:
        mask_sentence = flag_text.replace("[FLAG]", '[MASK]')

    probs, top_k_tokens = predict_char(tokenizer, model, source_sent, mask_sentence, max_length, k)
    probs = [prob.item() for prob in probs]
    return probs, top_k_tokens

def save_results(results, output_path):
    with open(output_path, 'w+', encoding='utf-8') as f_result:
        for r in results:
            tmp_write=' '.join(r).strip()
            f_result.write(tmp_write+ '\n')
good_dict=[line.strip().split(" ")[0].strip() for line in open("chinese_vocab.txt")]

def generate_bert(model, tokenizer, eval_file=None, max_length=128,sentences=None,mask_words=None,flag_text=None):
    with torch.no_grad():
        #sentences, mask_words = read_eval_dataset(eval_file)
        # sentences=["共有18支球队参加该届赛事的角逐，其中包括15支参加了的球队，以及从直接晋级的。"]
        # mask_words=["角逐"]
        

        results = []

        two_max_words=15
        #two_max_words=3
        for i in range(len(sentences)):
            len_word = len(mask_words[i])
            if len(sentences[i]) > int((max_length-3) / 2):
                start_index = sentences[i].index(mask_words[i])
                end_index = start_index + len_word
                sentences[i] = truncate(sentences[i], start_index, end_index, int((max_length-3) / 2))
            len_sentence = len(sentences[i])
            
            if len_word==4 or len_word==1:
                two_max_words=14

            other_max_words=50-two_max_words*3
            #other_max_words=10-two_max_words*2
            probs, substitutions = get_word_subs(tokenizer, model, sentences[i], mask_words[i], max_length,max_words=two_max_words,flag_text=flag_text)
            if len_word == 4:
                probs_other, idiom_substitutions = get_idiom_subs(tokenizer, model, sentences[i], mask_words[i], max_length,other_max_words,flag_text=flag_text)
                substitutions.extend(idiom_substitutions)
                probs.extend(probs_other)
            if len_word == 2:
                probs_other, one_char_word_substitutions = get_char_subs(tokenizer, model, sentences[i], mask_words[i], max_length,other_max_words,flag_text=flag_text)
                substitutions.extend(one_char_word_substitutions)
                probs.extend(probs_other)
            if len_word == 1:
                probs_other, one_char_word_substitutions = get_char_subs(tokenizer, model, sentences[i], mask_words[i], max_length,other_max_words,flag_text=flag_text)
                substitutions.extend(one_char_word_substitutions)
                probs.extend(probs_other)
            new_substitutions=substitutions
            new_probs=probs

            sorted_index=[new_probs.index(val) for val in sorted(new_probs,reverse=True)]
            new_new_substitutions=[new_substitutions[index1] for index1 in sorted_index]
            # new_substitutions=[word for word in substitutions if word in good_dict]
            # new_substitutions=[word for word in substitutions if word not in ["。","，","？","！","："]]
            results.append(new_new_substitutions)
        return results



# model_name_bert="bert-base-chinese"    
# tokenizer_bert = AutoTokenizer.from_pretrained(model_name_bert)
# model_bert = AutoModelForMaskedLM.from_pretrained(model_name_bert)
# results=generate(model_bert, tokenizer_bert, eval_file=None, max_length=128)
#print(results)
#finish_time=time.time()
#print("花费了",finish_time-begin_time)