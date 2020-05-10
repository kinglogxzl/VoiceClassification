# -*- coding: utf-8 -*-
'''
语音类别分析

'''
import codecs
import numpy as np
import pandas as pd
import os,sys
import wave

per_path = "个人扩展.txt"
ent_path = "企业扩展.txt"

def getlist(path, mode='caidan'):
    data = codecs.open(path, encoding="utf-8").readlines()
    result = []
    for line in data:
        line = line.strip()
        if (mode == 'kuozhan'):
            line = line.split('=')[0]
        result.append(line)
    return result


def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2] / len(word1)


def calc_distance(data):
    # 计算归一化编辑距离
    result = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            result[i][j] = edit_distance(data[i], data[j])
    result = pd.DataFrame(result)
    result.index = data
    result.columns = data
    return result

def output_similarity(data, pth, th=0.25):
    f = codecs.open(pth, "w", encoding="utf-8")
    for i in range(len(data.index)):
        line = data.iloc[i]
        line = line.sort_values(ascending=True)
        line = line[line.values<=th]
        content = data.index[i] + ' 相似阈值:' + str(th) + ' 相似个数:' + str(len(line)-1)  + '\n'
        if (len(line)==1):
            continue
        #line = line.to_string().split('\n')[1:]
        line = line.to_string().split('\n')
        content += " ".join(["%s" % (k) for k in line]) + "\n\n"
        f.write(content)

def output_list(data,pth):
    f = codecs.open(pth, "w", encoding="utf-8")
    for i,line in enumerate(data):
        f.write(line+'\t'+ str(i) +'\n')

# per_list = getlist(per_path,mode='caidan')
# output_list(per_list, 'per_label.txt')

# per_corr = calc_distance(per_list)
# output_similarity(per_corr, "per_corr.txt")
#
# ent_list = getlist(ent_path)
# output_list(ent_list, 'ent_label.txt')
# ent_corr = calc_distance(ent_list)
# output_similarity(ent_corr, "ent_corr.txt")


def time_count(paths):
    timecount = 0
    validd = 0
    sample_count = 0
    error_file = []
    for path in paths:
        file_list = os.listdir(path)
        for fi in file_list:
            if ('.wav' in fi):
                wav_path = path + fi
                #print (wav_path)
                try:
                    with wave.open(wav_path, 'rb') as f:
                        sample_count += 1
                        timecount += f.getparams().nframes / f.getparams().framerate
                        if (f.getparams().framerate==16000):
                            validd += 1
                        else:
                            print(f.getparams().framerate)
                            print (wav_path)
                except wave.Error:
                    error_file.append(wav_path)
    return timecount,error_file,sample_count,validd
path = ['/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/labled/jsnx20191111/','/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/labled/jy20191111/','/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/labled/rg20191111/']

# file_list = os.listdir(path[0])
# wav_path = path[0] + file_list[0]
# time_count = 0
# with wave.open('/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/labled/jy20191111/201901031408573124296.wav','rb') as f:
#     print (f.getparams().nframes)
#     print (f.getparams().framerate)
#     time_count += f.getparams().nframes / f.getparams().framerate

timecount,error_file,sample_count,validd = time_count(path)

print (len(error_file))
print ("sample_count", sample_count)
print ("time len(s)", timecount)
print (validd)

fi = open('error_file.txt','w')
for item in error_file:
    item = item[len('/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/'):]
    fi.write(item+'\n')