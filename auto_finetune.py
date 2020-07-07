# coding=utf-8
import shutil
import sys
import os
import time

sys.path.append('/home/qjsun/work/VoiceClassification')

from data.auto_finetune.label_analysis_autoTune import analysis_lable
from data.datautils import label_process
from data.process_data import preprocess_dataset

from network.train import train_network

PREPARE_DATA = False  # 是否有新数据需要预处理

if __name__ == '__main__':
    mark = 'merge'  # 标记不同时期的数据
    # key中文件存放wav文件名和具体类别，value表示wav文件的所在文件夹
    pre_dict = {"(geren)label.txt": "jsnx20191111", "zuixin.txt": "jsnx20191121-20200109",
                "test.txt": "jsnx20191121-20200109", "zonghe.txt": "jsnx20191121-20200109",
                "20200605.txt": "jsnx20191121-20200109", 'humancraft_label.txt': "humancraft_data"}
    # 存放key的路径
    pre_path = '/home/qjsun/work/VoiceClassification/data/auto_finetune/dir_label/'
    # 保存 完整wav文件路径和其类别说明 的路径
    out_label = '/home/qjsun/work/VoiceClassification/data/auto_finetune/label_' + mark + '.txt'
    # 生成类别分类html所在的路径
    html_path = '/home/qjsun/work/VoiceClassification/'

    # 原始wav文件存放所在文件夹
    source_path = '/data/voice/origin/'
    # 音频文件按类别存储输出路径
    data_path = '/data/voice/processed_' + mark + '/'

    # 新建数据集所在位置
    outpath = '/data/voice/logmeled64-' + mark + '/'
    if PREPARE_DATA:
        # 准备新数据
        # 1.计算label分布
        # analysis_lable(pre_dict=pre_dict, pre_path='./data/auto_finetune/dir_label/', out_label='./lable_test2.txt',html_path='./')
        analysis_lable(pre_dict=pre_dict, pre_path=pre_path, out_label=out_label, html_path=html_path)

        # 2.将原始数据分类，音频文件按类别存储
        label_process(file=out_label, data_path=data_path, source_path=source_path)

        # 3.处理音频文件，将梅尔频谱图以npy格式存储
        preprocess_dataset(inpath=data_path, outpath=outpath, resample=16000, mels=64)

    # 使用新数据集进行训练

    # 指定训练gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import argparse

    weight_save_path = "./weights/"
    weight_name = "weights_merge_7250.hdf5"
    weights = weight_save_path + weight_name
    classpath = outpath + "Train/"
    epochs = 100
    batch_size = 40
    val = 0.1
    maxper = 0
    reshape_x = 52
    parser = argparse.ArgumentParser(description="trains network using training dataset")
    parser.add_argument('-w', '--weights',  # nargs=1, type=argparse.FileType('r'),
                        help='weights file (in .hdf5)', default=weights)
    parser.add_argument('-c', '--classpath',  # type=argparse.string,
                        help='Train dataset directory with list of classes',
                        default=classpath)
    parser.add_argument('--epochs', default=epochs, type=int, help="Number of iterations to train for")
    parser.add_argument('--batch_size', default=batch_size, type=int, help="Number of clips to send to GPU at once")
    parser.add_argument('--val', default=val, type=float, help="Fraction of train to split off for validation")
    parser.add_argument("--tile", help="tile mono spectrograms 3 times for use with imagenet models",
                        action="store_true")
    parser.add_argument("--test", help="only test", action="store_true")
    parser.add_argument('-m', '--maxper', type=int, default=maxper, help="Max examples per class")
    parser.add_argument('-r', '--reshape', type=int, default=reshape_x, help="for LSTM Reshape Size")
    parser.add_argument('--tb_log', default='log', type=str, help="GPU used")

    drop_out_args = [[0.2, 0.2, 0.3, 0.3, 0.004], [0.2, 0.2, 0.3, 0.3, 0.007], [0.3, 0.3, 0.4, 0.4, 0.01], [0.4, 0.4, 0.5, 0.5, 0.02]]  # [0.2, 0.2, 0.3, 0.3, 0.004], [0.3, 0.3, 0.35, 0.35, 0.01],[0.4, 0.4, 0.5, 0.5, 0.02][0.4, 0.4, 0.5, 0.5, 0.04],[0.2, 0.2, 0.3, 0.3, 0.004], [0.2, 0.2, 0.3, 0.3, 0.007], [0.3, 0.3, 0.4, 0.4, 0.01],,[0.4, 0.4, 0.5, 0.5, 0.015],[0.4, 0.4, 0.5, 0.5, 0.03]
    index = 0
    args = parser.parse_args()
    for drop_out_arg in drop_out_args:
        train_network(weights_file=args.weights, classpath=args.classpath, epochs=args.epochs,
                      batch_size=args.batch_size, val_split=args.val, tile=args.tile, max_per_class=args.maxper,
                      only_test=args.test, reshape_x=args.reshape, drop_out_arg=drop_out_arg, tb_log=args.tb_log)
        for filename in os.listdir(weight_save_path):  # 用os.walk方法取得path路径下的文件夹路径，子文件夹名，所有文件名
            if filename == weight_name:
                index = index + 1
                new_name = filename.replace('.hdf5', '_' + str(index) + '.hdf5')  # 为文件赋予新名字
                shutil.copyfile(os.path.join(weight_save_path, filename),
                                os.path.join(weight_save_path, new_name))  # 复制并重命名文件
                print(filename, "copied as", new_name)
