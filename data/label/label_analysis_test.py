#
import re,shutil
def process_line(line):
    lst = re.split("[ \t]", line.strip())
    result = []
    for item in lst:
        if item == '':
            continue
        result.append(item)
    return result
# #%%
# file='/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/VoiceClassification/data/label/(个人银行标签)label.txt'
# data_path='../../labeled/data/'
# source_path = '/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/labled/jsnx20191111/'
# txt_cont = []
# f = open(file).readlines()
# print (len(f))
# valid = 0
# for line in f:
#     lst = process_line(line)
#     if (len(lst) == 3):  # 有效数据
#         valid += 1
#         target_path = data_path + lst[2]
#         txt_cont.append(lst)
#
# import matplotlib.pyplot as plt
# import numpy as np
# #改变绘图风格
# import seaborn as sns
# sns.set(color_codes=True)
#
#
#class_count


#%%
#给label文件增加目录
if __name__ == '__main__':
    import re,shutil,os,codecs
    txt_cont = []
    valid = 0
    pre_path = './' #'/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/VoiceClassification/data/label/'
    # label_file = os.listdir(pre_path)
    pre_dict = {"(个人银行标签)label.txt":"jsnx20191111","最新标注.txt":"jsnx20191121-20200109","test.txt":"jsnx20191121-20200109","综合.txt":"jsnx20191121-20200109"}
    for file in os.listdir(pre_path):
        if (not file in pre_dict.keys()):
            continue
        print (file)
        f = codecs.open(pre_path + file,'r',encoding='utf-8').readlines()
        print (len(f))
        for line in f:
            lst = process_line(line)
            if (len(lst) == 3):  # 有效数据
                valid += 1
                if (not file == "(个人银行标签)label.txt"):
                    lst[0] = pre_dict[file] + '/' + lst[0][:8] + '/' + lst[0]
                else:
                    lst[0] = pre_dict[file] + '/' + lst[0]
                txt_cont.append(lst)
    print (txt_cont[0])
    out_label = "./label_test.txt"
    f = codecs.open(out_label, 'w', encoding='utf-8')
    for line in txt_cont:
        cnt = ''
        for item in line:
            cnt += item + ' '
        cnt = cnt.strip() + '\n'
        f.write(cnt)

    classes = {}
    for item in txt_cont:
        if not item[2] in classes.keys():
            classes[item[2]] = 1
        else:
            classes[item[2]] += 1

    classes = sorted(classes.items(),key = lambda x:x[1],reverse = True)
    data = [['class','num']]
    sum = 0
    for item in classes:
        data.append([item[0],item[1]])
        sum += item[1]
    print (data)
    print ("sample sum:",sum)