#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/22 10:57
# @Author  : Chocolate
# @Site    : 
# @File    : tourism_sentiment_analysis.py
# @Software: PyCharm

# 这个是把结果集写入excel文件

from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
import os
import xlwt

#读取文件，文件读取函数
def read_file(filename):
    with  open(filename, 'r',encoding='utf-8')as f:
        text = f.read()
    return text

# 获取六种权值的词，根据要求返回list，这个函数是为了配合Django的views下的函数使用
def weighted_value(request):
    result_dict = []
    if request == "one":
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\most.txt")
        result_dict = read_file(r"degree_dict\most.txt")
    elif request == "two":
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\very.txt")
        result_dict = read_file(r"degree_dict\over.txt")
    elif request == "three":
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\more.txt")
        result_dict = read_file(r"degree_dict\very.txt")
    elif request == "four":
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\ish.txt")
        result_dict = read_file(r"degree_dict\more.txt")
    elif request == "five":
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\insufficiently.txt")
        result_dict = read_file(r"degree_dict\ish_insufficiently.txt")
    elif request == "six":
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\degree_dict\inverse.txt")
        result_dict = read_file(r"degree_dict\inverse.txt")
    elif request == 'posdict':
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\emotion_dict\pos_all_dict.txt")
        result_dict = read_file(r"emotion_dict\pos_all_dict.txt")
    elif request == 'negdict':
        # result_dict = read_file(r"E:\学习笔记\NLP学习\NLP code\情感分析3\emotion_dict\neg_all_dict.txt")
        result_dict = read_file(r"emotion_dict\neg_all_dict.txt")
    else:
        pass
    return result_dict

#文本分句
def cut_sentence(text):
    sentences = SentenceSplitter.split(text)
    sentence_list = [ w for w in sentences]
    return sentence_list

#文本分词
def tokenizer(sentence):
    #加载模型
    segmentor = Segmentor()  # 初始化实例
    # 加载模型
    # segmentor.load(r'E:\tool\python\Lib\site-packages\pyltp-0.2.1.dist-info\ltp_data\cws.model')
    segmentor.load(r'D:\CoModel\ltp_data_v3.4.0\ltp_data_v3.4.0\cws.model')
    # 产生分词，segment分词函数
    words = segmentor.segment(sentence)
    words = list(words)
    # 释放模型
    segmentor.release()
    return words

#去停用词函数
def del_stopwords(words):
    # 读取停用词表
    # stopwords = read_file(r"test_data\stopwords.txt")
    stopwords = read_file(r"stop_words\stopwords.txt")
    # 去除停用词后的句子
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words

#程度副词处理，对不同的程度副词给予不同的权重
def match_adverb(word,sentiment_value):
    # most
    if word in mostdict:
        sentiment_value *= 3
    # over
    elif word in overdict:
        sentiment_value *= 2.5
    # very
    elif word in verydict:
        sentiment_value *= 2
    # more
    elif word in moredict:
        sentiment_value *= 1.5
    # ish|insufficiently
    elif word in ish_insufficientdict:
        sentiment_value *= 0.5
    #否定词权重
    elif word in inversedict:
        sentiment_value *= -1
    else:
        sentiment_value *= 0.5
    return sentiment_value

#将数据写入文件中
# def write_data(filename,data):
#     with open(filename,'a',encoding='utf-8')as f:
#         f.write(str(data))

def run_score(contents):
    # 整篇游记的情感得分
    sentiment_scores = []
    # 文本分句
    sentences = cut_sentence(contents) # list类型
    # 对每一句话分词文本分词
    for sentence in sentences:
        # 分词
        words = tokenizer(sentence)
        # 去停用词
        seg_words = del_stopwords(words)
        # i，s 记录情感词和程度词出现的位置
        i = 0  # 记录扫描到的词位子
        s = 0  # 记录情感词的位置
        poscount = 0  # 记录积极情感词数目
        negcount = 0  # 记录消极情感词数目
        # 逐个查找情感词
        for word in seg_words:
            # 如果为积极词
            if word in posdict:
                poscount += 1  # 情感词数目加1
                # 在情感词前面寻找程度副词
                for w in seg_words[s:i]:
                    poscount = match_adverb(w, poscount)
                s = i + 1  # 记录情感词位置
            # 如果是消极情感词
            elif word in negdict:
                negcount += 1
                for w in seg_words[s:i]:
                    negcount = match_adverb(w, negcount)
                s = i + 1
            # 如果结尾为感叹号或者问号，表示句子结束，并且倒序查找感叹号前的情感词，权重+2
            # elif word =='!' or  word =='！' or word =='?' or word == '？':
            elif word == '!' or word == '?':
                for w2 in seg_words[::-1]:
                    # 如果为积极词，poscount+2
                    if w2 in posdict:
                        poscount += 2
                        break
                    # 如果是消极词，negcount+2
                    elif w2 in negdict:
                        negcount += 2
                        break
            i += 1  # 定位情感词的位置
        # 计算情感得分
        # 计算一句话的情感得分
        sentiment_score = poscount - negcount
        # 添加到整篇游记得分列表
        sentiment_scores.append(sentiment_score)
    # 计算整篇游记的得分
    sentiment_sum = 0
    for s in sentiment_scores:
        # 每一句话得分相加就是整篇游记的总得分
        sentiment_sum += s
    return sentiment_sum

if __name__ == '__main__':
    print("Processing...")
    print("reading sentiment dict .......")
    # 读取情感词典
    posdict = weighted_value('posdict')
    negdict = weighted_value('negdict')
    # 读取程度副词词典
    mostdict = weighted_value('one')  # 权值为3
    overdict = weighted_value('two')  # 权值为2.5
    verydict = weighted_value('three')  # 权值为2
    moredict = weighted_value('four')  # 权值为1.5
    ish_insufficientdict = weighted_value('five')  # 权值为0.5
    inversedict = weighted_value('six')  # 权值为-1

    # path_name = "D:/TourismData/旅游者层面/数据3.0/article"
    # path_name = "D:/test/article"
    path_name = "D:/TourismData/旅游者层面/数据3.0/未统计的article"
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象，相当于一个文件
    sheet = book.add_sheet('result_data', cell_overwrite_ok=True)  # 创建工作表
    col = ("游记标题", "游记链接", "情感分值", "情感标注")
    row = 1
    for i in range(0, 4):
        sheet.write(0, i, col[i])  # 列名
    for item in os.listdir(path_name):
        # 1. 读取文件
        text = read_file(os.path.join(path_name,item))
        text_list = text.split('\n')
        title = text_list[0]
        tille_url = text_list[1]
        contents = text_list[2]
        # 2. 计算分数
        scores = run_score(contents)
        # 3. 写入文件
        sheet.write(row, 0, title)
        sheet.write(row, 1, tille_url)
        sheet.write(row, 2, scores)
        if (scores > 0):
            sheet.write(row, 3, "积极")
        elif (scores < 0):
            sheet.write(row, 3, "消极")
        else:
            sheet.write(row, 3, "中性")
        row += 1
    book.save(r"d:/result_data.xls")

    print("succeed...")