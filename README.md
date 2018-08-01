# CIKM-AnalytiCup-2018

#### 题目描述:  
    本次算法竞赛是以聊天机器人中最常见的文本匹配算法为目标，通过语言适应技术构建跨语言的短文本匹配模型。在本次竞赛中，源语言为英语，目标语言为西班牙语。参赛选手可以根据主办方提供的数据，设计模型结构判断两个问句语义是否相同。最终，我们将在目标语言上测试模型的性能。  

#### 赛题详情:  
    https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.29b27257inkYar&raceId=231661

#### 运行环境
* 操作系统: Windows7  
* RAM: 16GB
* CPU: Intel(R) Core(TM) i7-4790K @ 4.00GHz     
* 显卡：GTX970 4GB  
* 语言：python3.6
* Python依赖工具:
    1.	Tensorflow-gpu == 1.8.0
    2.	scikit-learn == 0.19.1
    3.	numpy == 1.13.1
    4.	scipy == 1.0.0b1
    5.	gensim == 3.4.0
    6.	nltk == 3.2.5
    7.	tdqm == 4.20.0  
    8.  xgboost == 0.7  
    9.  lightgbm == 2.1.0
    10. fuzzywuzzy == 0.16.0  
    11. networkx == 2.1
    12. Levenshtein
    13. pattern == '2.6'  

#### 方案说明  
* ./Preprocessing:  
    i.Preprocess.py 数据预处理，利用编辑距离和回退模型的思想处理OOB数据  
    ii.Tokenizer.py 文本清洗，词干提取  
    iii.WordDict.py 建立word到index的映射  
    iv.Feature.py 特征提取，主要包括word_tfidf、char_tfidf、lsa相似度、d2v相似度、average_w2v相似度、句子长度(比)、ngram_jaccard_dis、ngram_dice_dis、fuzzywuzzy模糊距离、公共自序列长度等  
    v.PowerfulWord.py powerful_words特征  
    vi.GraphFeature.py 图特征(该部分特征导致线上线下不一致，最终未采用)  
    vii.Postprocess.py 利用图提取的规则修正最终结果以及rescale(最终未采用)  
    viii.GoogleTranslation.py 调用Google翻译翻译西班牙语料(最终未使用)
* ./Model:
    i.LexDecomp.py  
    implementation of the Answer Selection (AS) model proposed in the paper Sentence Similarity Learning by Lexical Decomposition and Composition, by (Wang et al., 2016).  
    论文地址：https://arxiv.org/pdf/1602.07019.pdf
    ii. AB-CNN.py  
    论文地址：https://arxiv.org/pdf/1512.05193.pdf  
    iii.Xgboost.py  Xgboost模型 
    iv.Embedding.py  获取Word2Vec及训练Doc2Vec  
    v.其他为父类或最终未采用的模型 
* ./Config:  
    配置信息及部分工具  
* ./Data:  
    原始数据  
* ./Cache:  
    中间缓存文件  
* ./Paper:  
    参考论文  
* ./Output:  
    输出结果  
* ./Save:  
    模型

#### 比赛成绩
* 第一阶段排名：14/1027
* 第二阶段排名：12/1027

#### 参考代码
    https://github.com/qqgeogor/kaggle-quora-solution-8th
    https://github.com/faneshion/MatchZoo
    






