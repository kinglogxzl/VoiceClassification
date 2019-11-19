# -*- coding: utf-8 -*-
'''
语音类别分析

'''
import codecs
import numpy as np
import pandas as pd

per_path = "个人菜单.txt"
ent_path = "企业菜单.txt"

def getlist(path):
    data = codecs.open(path, encoding="utf-8").readlines()
    result = []
    for line in data:
        line = line.strip()
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

per_list = getlist(per_path)
per_corr = calc_distance(per_list)
output_similarity(per_corr, "per_corr.txt")

ent_list = getlist(ent_path)
ent_corr = calc_distance(ent_list)
output_similarity(ent_corr, "ent_corr.txt")

