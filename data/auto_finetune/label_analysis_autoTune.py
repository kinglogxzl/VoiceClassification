#
import re, shutil


def process_line(line):
    lst = re.split("[ \t]", line.strip())
    result = []
    for item in lst:
        if item == '':
            continue
        result.append(item)
    return result


# #%%
# file='/Users/kinglog/Documents/learn/computer/研究生/融港语音识别/VoiceClassification/data/label/(geren)label.txt'
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
# class_count
def gen_echarts(data, html_path='./'):
    head = '''
    <!DOCTYPE html>
    <html style="height: 100%">
    <head>
        <meta charset="utf-8">
    </head>
    <body style="height: 100%; margin: 0">
        <div id="container" style="height: 100%"></div>
        <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
        <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts-gl/dist/echarts-gl.min.js"></script>
        <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts-stat/dist/ecStat.min.js"></script>
        <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/dist/extension/dataTool.min.js"></script>
        <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/dist/extension/bmap.min.js"></script>
        <script type="text/javascript">
        var dom = document.getElementById("container");
        var myChart = echarts.init(dom);
        var app = {};
        option = null;
        var canvas = document.createElement('canvas');
        var ctx = canvas.getContext('2d');
        canvas.width = 10;
        canvas.height = 1000;
        option = {
            dataset: {
                source: 
    '''
    # print(str(data))
    tail = '''
        },
        dataZoom: [
        {
            type: 'slider',
            show: true,
            yAxisIndex: [0],
            left: '93%',
            start: 0, //数据窗口范围的起始百分比
            end: 36
        },
        {
            type: 'inside',
            yAxisIndex: [0],
            start: 0,
            end: 36
        }
    ],
        tooltip: {},
        grid: {containLabel: true,
                height: 600
        },
        xAxis: {name: 'num'},
        yAxis: {type: 'category'},
        
        series: [
            {
                type: 'bar',
                encode: {
                    // Map the "amount" column to X axis.
                    x: 'num',
                    // Map the "product" column to Y axis
                    y: 'class'
                },
                label: {
                normal: {
                    position: 'right',
                    show: true
                }
            },
            }
        ]
    };
    ;
    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }
           </script>
       </body>
    </html>
    '''
    with open(html_path + 'class_distribution.html', 'w') as f:
        f.write(head + str(data) + tail)


def analysis_lable(pre_dict, pre_path='/home/qjsun/work/VoiceClassification/data/auto_finetune/dir_label/',
                   out_label='/home/qjsun/work/VoiceClassification/data/auto_finetune/label_test.txt', html_path='./'):
    import re, shutil, os, codecs
    txt_cont = []
    label_file = os.listdir(pre_path)
    wav_name_set = set()
    for file in label_file:
        if (not file in pre_dict.keys()):
            continue
        print(file)
        f = codecs.open(pre_path + file, 'r', encoding='utf-8').readlines()
        # print(len(f))
        for line in f:
            lst = process_line(line)
            if (file == 'humancraft_label.txt'):
                if (len(lst) == 2):  # 有效数据
                    lst.append(lst[1])
                    if(not lst[0] in wav_name_set):
                        wav_name_set.add(lst[0])
                        lst[0] = pre_dict[file] + '/' + lst[0]
                        txt_cont.append(lst)
            else:
                if (len(lst) == 3):  # 有效数据
                    # valid += 1
                    if (not lst[0] in wav_name_set):
                        wav_name_set.add(lst[0])
                        if (not file == "(geren)label.txt"):
                            lst[0] = pre_dict[file] + '/' + lst[0][:8] + '/' + lst[0]
                        else:
                            lst[0] = pre_dict[file] + '/' + lst[0]
                        txt_cont.append(lst)
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

    classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
    data = [['class', 'num']]
    for item in classes:
        data.append([item[0], item[1]])
    gen_echarts(data, html_path)


# %%
# 给label文件增加目录
if __name__ == '__main__':
    pre_dict = {"(geren)label.txt": "jsnx20191111", "zuixin.txt": "jsnx20191121-20200109",
                "test.txt": "jsnx20191121-20200109", "zonghe.txt": "jsnx20191121-20200109",
                'humancraft_label.txt': "humancraft_data"}
    analysis_lable(pre_dict=pre_dict, pre_path='./dir_label', out_label='./lable_test2.txt')
