# Chinese Lexical Substitution: Dataset and Method
Existing lexical substitution (LS) benchmarks were collected by asking human annotators to think of substitutes from memory, resulting in benchmarks with limited coverage and relatively small scales. To overcome this problem, we propose a novel annotation method to construct an LS dataset based on human and machine collaboration. Based on our annotation method, we construct the first Chinese LS dataset CHNLS which consists of 33,695 instances and 144,708 substitutes, covering three text genres (News, Novel, and Wikipedia). Specifically, we first combine four unsupervised LS methods as an ensemble method to generate the candidate substitutes, and then let human annotators judge these candidates or add new ones. This collaborative process combines the diversity of machine-generated substitutes with the expertise of human annotators. Experimental results that the ensemble method outperforms other LS methods. To our best knowledge, this is the first study for the Chinese LS task\footnote{Code and data are available at https:}. 
![](overview.jpg)

# Dataset Release
Our dataset include three domains: wiki, novel, and news. We split the whole dataset  into train (80%), valid (10%), test (10%) set. The train/valid/test sets in Wiki, News, and Novel have 8,425/1,065/888, 9,472/1,169/1,110, and 9,379/1,080/11,07 instances, respectively. The dataset could be found in dataset/release/

### Test dataset
To calculate the corresponding metrics, we have restructured the test dataset following [SemEval07](https://aclanthology.org/S07-1009.pdf). This dataset can be found in the dataset/Chinese_LS/ directory.

# Install Instruction
### Transformers
Our method is build on transformers, based on custom modification of scripts. You need first follow the commands to install transformers
cd transformers-4.20.1_phrase_ahead_chinese
pip install -e .

### Other dependencies

pip install -r requirements.txt

# Models 

 [Embedding Model](https://github.com/Embedding/Chinese-Word-Vectors), need placed in models/

[DICT](https://pan.baidu.com/share/link?shareid=2858555949&uk=2738088569),haved placed in dataset/dict/

[Paraphraser](https://drive.google.com/file/d/17-i8xwkBqfLcLygXywslPpEuMs4ANKOA/view?usp=sharing), need placed in bart-base-chinese/


[BERT](https://huggingface.co/bert-base-chinese), placed in bert-base-chinese/



# Run example for news category 

### Dict
```
python test.19.news.test.dictonary.py
```

### Embedding 
```
python test.19.news.test.embed.py
```
### ParaLS

```
python test.19.news.test.bart.ahead.py
```

### BERT
```
python test.19.news.test.bert.py
```

### Our Emsemble method
```
python test.19.news.sort.number.py
```

# Results
![](results.jpg)


# Citation
```
@inproceedings{qiangchls,
title={Chinese Lexical Substitution: Dataset and Method},
author={Jipeng Qiang and Kang Liu and Yin Li and Yun Li and Yunhao Yuan and Yi Zhu
Xiaoye Ouyang and Xiaocheng Hu},
booktitle={EMNLP},
year={2023}
}
```

# Contact

If you have any question. Please contact yzunlplk@163.com.
