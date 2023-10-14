def read_dict(dict_path):
    dict = []
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            entry = line[9:].strip().split(' ')
            if entry:
                dict.append(entry)
    return dict
#difficult_words=["晋级"] 
#dict1=read_dict("dataset/dict/HIT-dict=.txt")
def generate_dict(dict1,difficult_words):
    final_substitution_words=[]
    for difficult_word in difficult_words:
        isFound = False
        substitution_words = []
        for entry in dict1:
            if difficult_word in entry:
                isFound = True
                for word in entry:
                    substitution_words.append(word)
        # if (isFound == False):
        #     substitution_words.append('NULL')
        final_substitution_words.append(substitution_words)
        #print(substitution_words)
    return final_substitution_words
    
#generate_dict(dict1,difficult_words)
import re
def process_tmp_result(tmp_result_list_str):
    tmp_result_list_str=tmp_result_list_str.lstrip("\"")
    tmp_result_list_str=tmp_result_list_str.rstrip("\"")

    tmp_result_list_str=tmp_result_list_str.replace("。","")
    tmp_result_list_str=tmp_result_list_str.replace(",","、")
    tmp_result_list_str=tmp_result_list_str.replace("，","、")
    tmp_result_list_str=tmp_result_list_str.replace("\n","、")

    # if("1." in tmp_result_list_str):
    #     import pdb
    #     pdb.set_trace()
    if("1." in tmp_result_list_str):
        tmp_result_list_str=re.sub("([1-9]|10)\.", "", tmp_result_list_str, count=0, flags=0)
    tmp_result_list_str=tmp_result_list_str.replace(".","")

    return tmp_result_list_str.split("、")


def generate_chatgpt(path1,path2):
    import pickle
    final_substitution_words=[]
    
    with open(path1, "rb") as f2:
        dict_list = []
        while True:
            try:
                dict_obj = pickle.load(f2)
                dict_list.append(dict_obj)
            except EOFError:
                break
    number1=[line.strip().split("\t")[1] for line in open(path2)]
    index1=0
    number2index={}
    assert len(number1)==len(dict_list)  
    for dict_obj in dict_list:
        tmp_result_list_str=dict_obj["content"].lower().strip()
        tmp_result_list=process_tmp_result(tmp_result_list_str)
        final_substitution_words.append(tmp_result_list)
        number2index[number1[index1]]=index1
        index1+=1

    return final_substitution_words,number2index



def generate_chatgpt(path1,path2):
    import pickle
    final_substitution_words=[]
    
    with open(path1, "rb") as f2:
        dict_list = []
        while True:
            try:
                dict_obj = pickle.load(f2)
                dict_list.append(dict_obj)
            except EOFError:
                break
    number1=[line.strip().split("\t")[1] for line in open(path2)]
    index1=0
    number2index={}
    import pdb
    pdb.set_trace()
    assert len(number1)==len(dict_list)  
    for dict_obj in dict_list:
        tmp_result_list_str=dict_obj["content"].lower().strip()
        tmp_result_list=process_tmp_result(tmp_result_list_str)
        final_substitution_words.append(tmp_result_list)
        number2index[number1[index1]]=index1
        index1+=1

    return final_substitution_words,number2index


def generate_chatgpt_2(path1,path2):
    import pickle
    final_substitution_words=[]
    
    # with open(path1, "rb") as f2:
    #     dict_list = []
    #     while True:
    #         try:
    #             dict_obj = pickle.load(f2)
    #             dict_list.append(dict_obj)
    #         except EOFError:
    #             break

    dict_list=open(path1).readlines()

    number1=[line.strip().split("\t")[1] for line in open(path2)]
    index1=0
    number2index={}
    assert len(number1)==len(dict_list)  
    for dict_obj in dict_list:
        #tmp_result_list_str=dict_obj["content"].lower().strip()
        tmp_result_list_str=dict_obj.strip().split("|||")[0]
        tmp_result_list=process_tmp_result(tmp_result_list_str)
        final_substitution_words.append(tmp_result_list)
        number2index[number1[index1]]=index1
        index1+=1

    return final_substitution_words,number2index


import json
def generate_chatgpt_3(path1,path2):
    import pickle
    final_substitution_words=[]


    dict_list=[json.loads(line.strip()) for line in open(path1)]

    number1=[line.strip().split("\t")[1] for line in open(path2)]
    index1=0
    number2index={}
    assert len(number1)==len(dict_list)  
    for dict_obj in dict_list:
        #tmp_result_list_str=dict_obj["content"].lower().strip()
        try:
            tmp_result_list_str=dict_obj["content"].lower().strip()
        except:
            import pdb
            pdb.set_trace()
        tmp_result_list=process_tmp_result(tmp_result_list_str)
        final_substitution_words.append(tmp_result_list)
        number2index[number1[index1]]=index1
        index1+=1

    return final_substitution_words,number2index



