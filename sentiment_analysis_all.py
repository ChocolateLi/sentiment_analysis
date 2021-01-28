# -*- coding:utf8 -*-
#大明王
import pyltp
from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
import numpy as np

#读取文件，文件读取函数
def read_file(filename):
    with  open(filename, 'r',encoding='utf-8')as f:
        text = f.read()
        #返回list类型数据
        text = text.split('\n')
    return text

#将数据写入文件中
def write_data(filename,data):
    with open(filename,'a',encoding='utf-8')as f:
        f.write(str(data))

#文本分句
def cut_sentence(text):
    sentences = SentenceSplitter.split(text)
    sentence_list = [ w for w in sentences]
    return sentence_list

#文本分词
def tokenize(sentence):
    #加载模型
    segmentor = Segmentor()  # 初始化实例
    # 加载模型
    segmentor.load(r'E:\tool\python\Lib\site-packages\pyltp-0.2.1.dist-info\ltp_data\cws.model')
    # 产生分词，segment分词函数
    words = segmentor.segment(sentence)
    words = list(words)
    # 释放模型
    segmentor.release()
    return words

#词性标注
def postagger(words):
    # 初始化实例
    postagger = Postagger()
    # 加载模型
    postagger.load(r'E:\tool\python\Lib\site-packages\pyltp-0.2.1.dist-info\ltp_data\pos.model')
    # 词性标注
    postags = postagger.postag(words)
    # 释放模型
    postagger.release()
    #返回list
    postags = [i for i in postags]
    return postags

# 分词，词性标注，词和词性构成一个元组
def intergrad_word(words,postags):
    #拉链算法，两两匹配
    pos_list = zip(words,postags)
    pos_list = [ w for w in pos_list]
    return pos_list

#去停用词函数
def del_stopwords(words):
    # 读取停用词表
    stopwords = read_file(r"test_data\stopwords.txt")
    # 去除停用词后的句子
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words

# 获取六种权值的词，根据要求返回list，这个函数是为了配合Django的views下的函数使用
def weighted_value(request):
    result_dict = []
    if request == "one":
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\most.txt")
    elif request == "two":
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\very.txt")
    elif request == "three":
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\more.txt")
    elif request == "four":
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\ish.txt")
    elif request == "five":
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\insufficiently.txt")
    elif request == "six":
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\inverse.txt")
    elif request == 'posdict':
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\emotion_dict\pos_all_dict.txt")
    elif request == 'negdict':
        result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\emotion_dict\neg_all_dict.txt")
    else:
        pass
    return result_dict


print("reading sentiment dict .......")
# 读取情感词典
posdict = weighted_value('posdict')
negdict = weighted_value('negdict')
# 读取程度副词词典
mostdict = weighted_value('one')# 权值为4
verydict = weighted_value('two')# 权值为3
moredict = weighted_value('three')# 权值为2
ishdict = weighted_value('four')# 权值为0.5
insufficientdict = weighted_value('five') # 权值为0.25
inversedict = weighted_value('six')# 权值为-1

#程度副词处理，对不同的程度副词给予不同的权重
def match_adverb(word,sentiment_value):
    #最高级权重为
    if word in mostdict:
        sentiment_value *= 8
    #比较级权重
    elif word in verydict:
        sentiment_value *= 6
    #一般级权重
    elif word in moredict:
        sentiment_value *= 4
    #轻微程度词权重
    elif word in ishdict:
        sentiment_value *= 2
    #相对程度词权重
    elif word in insufficientdict:
        sentiment_value *= 0.5
    #否定词权重
    elif word in inversedict:
        sentiment_value *= -1
    else:
        sentiment_value *= 1
    return sentiment_value

#对每一条微博打分
def single_sentiment_score(text_sent):
    sentiment_scores = []
    #对单条微博分句
    sentences = cut_sentence(text_sent)
    for sent in sentences:
        #查看分句结果
        # print('分句：',sent)
        #分词
        words = tokenize(sent)
        seg_words = del_stopwords(words)
        #i，s 记录情感词和程度词出现的位置
        i = 0   #记录扫描到的词位子
        s = 0   #记录情感词的位置
        poscount = 0 #记录积极情感词数目
        negcount = 0 #记录消极情感词数目
        #逐个查找情感词
        for word in seg_words:
            #如果为积极词
            if word in posdict:
                poscount += 1  #情感词数目加1
            #在情感词前面寻找程度副词
                for w in seg_words[s:i]:
                    poscount = match_adverb(w,poscount)
                s = i+1 #记录情感词位置
            # 如果是消极情感词
            elif word in negdict:
                negcount +=1
                for w in seg_words[s:i]:
                    negcount = match_adverb(w,negcount)
                s = i+1
            #如果结尾为感叹号或者问号，表示句子结束，并且倒序查找感叹号前的情感词，权重+4
            elif word =='!' or  word =='！' or word =='?' or word == '？':
                for w2 in seg_words[::-1]:
                    #如果为积极词，poscount+2
                    if w2 in posdict:
                        poscount += 4
                        break
                    #如果是消极词，negcount+2
                    elif w2 in negdict:
                        negcount += 4
                        break
            i += 1 #定位情感词的位置
        #计算情感值
        sentiment_score = poscount - negcount
        sentiment_scores.append(sentiment_score)
        #查看每一句的情感值
        # print('分句分值：',sentiment_score)
    sentiment_sum = 0
    for s in sentiment_scores:
        #计算出一条微博的总得分
        sentiment_sum +=s
    return sentiment_sum

# 分析test_data.txt 中的所有微博，返回一个列表，列表中元素为（分值，微博）元组
def run_score(contents):
    # 待处理数据
    scores_list = []
    for content in contents:
        if content !='':
            score = single_sentiment_score(content)  # 对每条微博调用函数求得打分
            scores_list.append((score, content)) # 形成（分数，微博）元组
    return scores_list

#主程序
if __name__ == '__main__':
    print('Processing........')
    #测试
    # sentence = '要怎么说呢! 我需要的恋爱不是现在的样子, 希望是能互相鼓励的勉励, 你现在的样子让我觉得很困惑。 你到底能不能陪我一直走下去, 你是否有决心?是否你看不惯我?你是可以随意的生活,但是我的未来我耽误不起！'
    # sentence = '转有用吗？这个事本来就是要全社会共同努力的，公交公司有没有培训到位？公交车上地铁站内有没有放足够的宣传标语？我现在转一下微博，没有多大的意义。'
    sentences = read_file(r'test_data\微博.txt')
    scores = run_score(sentences)
    #人工标注情感词典
    man_sentiment = read_file(r'test_data\人工情感标注.txt')
    al_sentiment = []
    for score in scores:
        print('情感分值：',score[0])
        if score[0] > 0:
            print('情感倾向：消极')
            s = '积极'
        elif score[0] == 0:
            print('情感倾向：中性')
            s = '中性'
        else:
            print('情感倾向：积极')
            s = '消极'
        al_sentiment.append(s)
        print('情感分析文本：',score[1])
    i = 0
    #写入文件中
    filename = r'result_data\result_data.txt'
    for score in scores:
        write_data(filename, '情感分析文本：{}'.format(str(score[1]))+'\n') #写入情感分析文本
        write_data(filename,'情感分值：{}'.format(str(score[0]))+'\n') #写入情感分值
        write_data(filename,'人工标注情感：{}'.format(str(man_sentiment[i]))+'\n') #写入人工标注情感
        write_data(filename, '机器情感标注：{}'.format(str(al_sentiment[i]))+'\n') #写入机器情感标注
        write_data(filename,'\n')
        i +=1
    print('succeed.......')