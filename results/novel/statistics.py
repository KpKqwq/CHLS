
dict_output=open("dict/novel.out.embed.0.1.oot").readlines()
# dict_output=open("embed/novel.out.embed.0.1.oot").readlines()
# dict_output=open("bert/novel.out.embed.0.1.oot").readlines()
# dict_output=open("sort.bart.ahead/novel.out.embed.0.1.oot").readlines()
# dict_output=open("sort.number/all/novel.out.embed.0.1.oot").readlines()

# dict_len=set()
# dict_len.add(len(dict_output))
# dict_len.add(len(embed_output))
# dict_len.add(len(bert_output))
# dict_len.add(len(bart_output))
# dict_len.add(len(hybrid_output))


# if(len(dict_len)>1):
#     print("wrong with the length")
#     import pdb
#     pdb.set_trace()

dict_file=open("dict/statistic","w+")
# dict_file=open("embed/statistic","w+")
# dict_file=open("bert/statistic","w+")
# dict_file=open("sort.bart.ahead/statistic","w+")
# dict_file=open("sort.number/all/statistic","w+")

# embed_file=open("embed/statistic","w+")
# bert_file=open("bert/statistic","w+")
# bart_file=open("sort.bart.ahead/statistic","w+")
# hybrid_file=open("sort.number/all/statistic","w+")

labels_dict={}
for line in open("/home/yz/liukang/liukang/PhraseSimplification/dataset/Chinese_LS/novel_test_gold.txt"):
    key1=line.split("::")[0].strip()
    val1=line.split("::")[1].strip()
    labels_dict[key1]=val1

from reader import Reader_lexical
reader = Reader_lexical()
eval_dir="../../dataset/Chinese_LS/novel_test_processed.txt"
reader.create_feature(eval_dir)
count_gen=-1
from tqdm import tqdm
for main_word in tqdm(reader.words_candidate):
    for instance in reader.words_candidate[main_word]:
        for context in reader.words_candidate[main_word][instance]:
            count_gen+=1
            dict_line=dict_output[count_gen].strip()
            # embed_line=embed_output[count_gen].strip()
            # bert_line=bert_output[count_gen].strip()
            # bart_line=bart_output[count_gen].strip()
            # hybrid_line=hybrid_output[count_gen].strip()


            try:
                dict_output_line=dict_line.split(":::")[1].strip()
            except:
                import pdb
                pdb.set_trace()
            dict_lst=dict_output_line.split(";")


            text = context[1]
            original_text = text
            original_words = text.split(' ')
            index_word = int(context[2])
            target_word = text.split(' ')[index_word]
            target_pos = main_word.split('.')[-1]
            target_lemma=target_word.lower().strip()

            label_key=main_word.strip()+" "+instance.strip()

            labels=labels_dict[label_key][:-1]
            label_lst=[tmp_word1.split()[0] for tmp_word1 in labels.split(";")]
            labels=";".join(label_lst)
            # if count_gen==6:
            #     import pdb
            #     pdb.set_trace()
            tp_list=[word for word in dict_lst if word in label_lst]
            #dict_file.write(original_text+"|||"+target_word+"|||"+labels+"|||"+dict_output_line+"|||"+";".join(tp_list)+"\n")
            dict_file.write(labels+"|||"+dict_output_line+"|||"+";".join(tp_list)+"\n")
            

dict_file.close()