#! /usr/bin/env python3

from __future__ import print_function
import sys

sys.path.append('/home/qjsun/work/VoiceClassification')
print(sys.path)
print(sys.version)
# import numpy as np
from data.datautils import *
# from keras.callbacks import ModelCheckpoint #,EarlyStopping
import os
# from os.path import isfile
# from timeit import default_timer as timer
# from network.muti_gpu import MultiGPUModelCheckpoint
from network.mixup_generator import MixupGenerator
from network.models import *
from keras.callbacks import TensorBoard


# import math


def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/",
                  epochs=50, batch_size=20, val_split=0.2, tile=False, max_per_class=0, only_test=False,
                  reshape_x=52, drop_out_arg=[], tb_log='log'):
    np.random.seed(1)  # 初始化随机种子

    # Get the data
    X_train, Y_train, paths_train, class_names = build_dataset(path=classpath,
                                                               batch_size=batch_size, tile=tile,
                                                               max_per_class=max_per_class)
    X_test, Y_test, paths_test, class_names_test = build_dataset(path=classpath + "../Test/", batch_size=batch_size,
                                                                 tile=tile, max_per_class=int(max_per_class / 8 * 2))
    print("============================LOAD DATASET============================")
    # if classpath=="/data/voice/logmeled/Train/":
    #     reshape_x = 39
    # elif classpath=="/data/voice/logmeled64_test/Train/":
    #     reshape_x = 52
    # Instantiate the model
    model, serial_model = setup_model(X_train, class_names, weights_file=weights_file, reshape_x=reshape_x,
                                      drop_out_arg=drop_out_arg)
    if os.path.exists(tb_log):
        files = glob.glob(os.path.join(tb_log, '*.ubuntu-sever'))
        for f in files:
            os.remove(f)
    if not only_test:
        save_best_only = True

        split_index = int(X_train.shape[0] * (1 - val_split))
        # X_val, Y_val = X_train[split_index:], Y_train[split_index:]
        # X_train, Y_train = X_train[:split_index - 1], Y_train[:split_index - 1]

        checkpointer = MultiGPUModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=save_best_only,
                                               serial_model=serial_model, period=1, class_names=class_names)

        steps_per_epoch = X_train.shape[0] // batch_size
        if False and ((len(class_names) > 2) or (steps_per_epoch > 1)):
            training_generator = MixupGenerator(X_train, Y_train, batch_size=batch_size, alpha=0.25)()
            model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch,
                                epochs=epochs, shuffle=True,
                                verbose=1, callbacks=[checkpointer, TensorBoard(log_dir=tb_log)],
                                validation_data=(X_test, Y_test))
        else:
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                      verbose=1, callbacks=[checkpointer, TensorBoard(log_dir=tb_log)],  # validation_split=val_split)
                      validation_data=(X_test, Y_test))
            # model.save('weights-last.hdf5')
    # overwrite text file class_names.txt  - does not put a newline after last class name
    with open('class_names.txt', 'w') as outfile:
        outfile.write("\n".join(class_names))

    # Score the model against Test dataset
    # X_test, Y_test, paths_test, class_names_test = build_dataset(path=classpath + "../Test/", tile=tile)
    print("Start Test...")
    assert (class_names == class_names_test)
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    predict = model.predict(X_test)
    score = predict
    predict_re = np.argsort(predict, axis=1)
    # predict_re = predict_re[0][::-1][:5]
    predict = np.argmax(predict, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    class_acc = {}
    for item in Y_test:
        if item not in class_acc.keys():
            class_acc[item] = 1
        else:
            class_acc[item] += 1
    correct_class = {item: 0 for item in class_acc.keys()}
    correct_sum = 0
    import codecs
    c_num_to_name = {}
    labelf_name = '/home/qjsun/work/VoiceClassification/data/per_label.txt'
    f = codecs.open(labelf_name, 'r', encoding='utf-8')
    if only_test:
        for line in f.readlines():
            item = line.strip().split('\t')
            c_num_to_name[item[1]] = item[0]
        for i, item in enumerate(predict_re):
            item = item[::-1][:5]
            if item[0] == Y_test[i]:
                correct_class[item[0]] += 1
                correct_sum += 1
            else:
                with open("./false_data.txt", 'a', encoding="utf-8") as f:
                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(paths_test[i], c_num_to_name[class_names[Y_test[i]]],
                                                                  c_num_to_name[class_names[item[0]]],score[i][item[0]],
                                                                  c_num_to_name[class_names[item[1]]],score[i][item[1]],
                                                                  c_num_to_name[class_names[item[2]]],score[i][item[2]],
                                                                  c_num_to_name[class_names[item[3]]],score[i][item[3]],
                                                                  c_num_to_name[class_names[item[4]]],score[i][item[4]]))
    # 按照正确数量排序
    correct_class = sorted(correct_class.items(), key=lambda item: item[1], reverse=True)
    import codecs
    c_num_to_name = {}
    labelf_name = '/home/qjsun/work/VoiceClassification/data/per_label.txt'
    f = codecs.open(labelf_name, 'r', encoding='utf-8')
    for line in f.readlines():
        item = line.strip().split('\t')
        c_num_to_name[item[1]] = item[0]
    print("=============总准确率=============")
    print("总样本量: ", len(Y_test), "\t准确率: ", round(correct_sum / len(Y_test), 4))
    print("=============各类别回归率=============")
    for key, value in correct_class:
        if class_names[key] not in c_num_to_name.keys():
            cla_name = '*****标识错误*****'
        else:
            cla_name = c_num_to_name[class_names[key]]
        print("类别: ", class_names[key], "\t样本量: ", class_acc[key], "\t回归率: ", round(value / class_acc[key], 4),
              '\t类别名:', cla_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="trains network using training dataset")
    parser.add_argument('-w', '--weights',  # nargs=1, type=argparse.FileType('r'),
                        help='weights file (in .hdf5)', default="newdata_weights.hdf5")
    parser.add_argument('-c', '--classpath',  # type=argparse.string,
                        help='Train dataset directory with list of classes',
                        default="/data/voice/logmeled64_test/Train/")
    parser.add_argument('--epochs', default=50, type=int, help="Number of iterations to train for")
    parser.add_argument('--batch_size', default=40, type=int, help="Number of clips to send to GPU at once")
    parser.add_argument('--val', default=0, type=float, help="Fraction of train to split off for validation")
    parser.add_argument("--tile", help="tile mono spectrograms 3 times for use with imagenet models",
                        action="store_true")
    parser.add_argument("--test", help="only test", action="store_true")
    parser.add_argument('-m', '--maxper', type=int, default=0, help="Max examples per class")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    train_network(weights_file=args.weights, classpath=args.classpath, epochs=args.epochs, batch_size=args.batch_size,
                  val_split=args.val, tile=args.tile, max_per_class=args.maxper, only_test=args.test)
