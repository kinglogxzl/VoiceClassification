'''
datautils.py:  Just some routines that we use for moving data around
'''
from __future__ import print_function
import numpy as np
import librosa
import os
from os.path import isfile, splitext
from imageio import imread, imwrite
import glob
from skimage import img_as_ubyte
from random import shuffle
import re, shutil
from tqdm import tqdm


def listdir_nohidden(path, subdirs_only=False, skip_csv=True):
    '''
    ignore hidden files. call should be inside list().  subdirs_only means it ignores regular files
    '''
    for f in os.listdir(path):
        if not f.startswith('.'):  # this skips the hidden
            if ((False == subdirs_only) or (os.path.isdir(path + "/" + f))):
                if ('.csv' == os.path.splitext(f)[1]) and (skip_csv):
                    pass
                else:
                    yield f


# class names are subdirectory names in Preproc/ directory
def get_class_names(path="Preproc/Train/", sort=True):
    if (sort):
        class_names = sorted(
            list(listdir_nohidden(path, subdirs_only=True)))  # sorted alphabetically for consistency with "ls" command
    else:
        class_names = listdir_nohidden(path)  # not in same order as "ls", because Python
    return class_names


def get_total_files(class_names, path="Preproc/Train/"):
    '''
    获取音频文件数量
    :param class_names:
    :param path:
    :return:
    '''
    sum_total = 0
    for subdir in class_names:
        files = os.listdir(path + subdir)
        n_files = len(files)
        sum_total += n_files
    return sum_total


def scale_to_uint8(float_img):
    # out_img = 255*(float_img - np.min(float_img))/np.ptp(float_img).astype(np.uint8)
    out_img = img_as_ubyte((float_img - np.min(float_img)) / np.ptp(float_img))
    return out_img


def save_melgram(outfile, melgram, out_format='npz'):
    channels = melgram.shape[3]
    melgram = melgram.astype(np.float16)
    if (('jpeg' == out_format) or ('png' == out_format)) and (channels <= 4):
        melgram = np.squeeze(melgram)  # squeeze gets rid of dimensions of batch_size 1
        # melgram = np.moveaxis(melgram, 1, 3).squeeze()      # we use the 'channels_first' in tensorflow, but images have channels_first. squeeze removes unit-size axes
        melgram = np.flip(melgram, 0)  # flip spectrogram image right-side-up before saving, for viewing
        if (2 == channels):  # special case: 1=greyscale, 3=RGB, 4=RGBA, ..no 2.  so...?
            # pad a channel of zeros (for blue) and you'll just be stuck with it forever. so channels will =3
            # TODO: this is SLOWWW
            b = np.zeros((melgram.shape[0], melgram.shape[1], 3))  # 3-channel array of zeros
            b[:, :, :-1] = melgram  # fill the zeros on the 1st 2 channels
            imwrite(outfile, scale_to_uint8(b), format=out_format)
        else:
            imwrite(outfile, scale_to_uint8(melgram), format=out_format)
    elif ('npy' == out_format):
        np.savez(outfile, melgram=melgram)
    else:
        np.savez_compressed(outfile, melgram=melgram)  # default is compressed npz file
    return


def load_audio(audio_path, mono=None, sr=None, convertOSXaliases=True):  # wrapper for librosa.load

    signal, sr = librosa.load(audio_path, mono=mono, sr=sr)
    # except NoBackendError as e:
    #     if ('Darwin' == platform.system()):   # handle OS X alias files gracefully
    #         source = resolve_osx_alias(audio_path, convert=convertOSXaliases, already_checked_os=True) # convert to symlinks for next time
    #         try:
    #             signal, sr = librosa.load(source, mono=mono, sr=sr)
    #         except NoBackendError as e:
    #             print("\n*** ERROR: Could not open audio file {}".format(audio_path),"\n",flush=True)
    #             raise e
    #     else:
    #         print("\n*** ERROR: Could not open audio file {}".format(audio_path),"\n",flush=True)
    #         raise e
    return signal, sr


def load_melgram(file_path):
    # auto-detect load method based on filename extension
    name, extension = os.path.splitext(file_path)
    if ('.npy' == extension):
        melgram = np.load(file_path)
    elif ('.npz' == extension):  # compressed npz file (preferred)
        with np.load(file_path) as data:
            melgram = data['melgram']
    elif ('.png' == extension) or ('.jpeg' == extension):
        arr = imread(file_path)
        melgram = np.reshape(arr, (1, arr.shape[0], arr.shape[1], 1))  # convert 2-d image
        melgram = np.flip(melgram,
                          0)  # we save images 'rightside up' but librosa internally presents them 'upside down'
    else:
        print("load_melgram: Error: unrecognized file extension '", extension, "' for file ", file_path, sep="")
    # print("melgram.shape = ",melgram.shape)
    return melgram


def get_sample_dimensions(class_names, path='Preproc/Train/'):
    '''
    加载第一个音频文件，采用其shape
    :param class_names:
    :param path:
    :return:
    '''
    classname = class_names[0]
    audio_path = path + classname + '/'
    infilename = os.listdir(audio_path)[0]
    melgram = load_melgram(audio_path + infilename)
    print("   get_sample_dimensions: " + infilename + ": melgram.shape = ", melgram.shape)
    return melgram.shape


def encode_class(class_name, class_names, label_smoothing=0.005):
    # makes a "one-hot" vector for each class name called
    #  label_smoothing is a parameter to make the training more robust to mislabeled data
    try:
        idx = class_names.index(class_name)
        num_classes = len(class_names)
        vec = np.zeros(num_classes)
        vec[idx] = 1

        if label_smoothing > 0:
            vec = vec * (1 - label_smoothing) + label_smoothing / num_classes
        return vec
    except ValueError:
        return None


def decode_class(vec, class_names):  # generates a number from the one-hot vector
    return int(np.argmax(vec))


def shuffle_XY_paths(X, Y, paths):  # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0])
    # print("shuffle_XY_paths: Y.shape[0], len(paths) = ",Y.shape[0], len(paths))
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths[:]
    for i in range(len(idx)):
        newX[i] = X[idx[i], :, :]
        newY[i] = Y[idx[i], :]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths


def make_melgram(mono_sig, sr, n_mels=128):  # @keunwoochoi upgraded form 96 to 128 mel bins in kapre
    # melgram = librosa.logamplitude(librosa.feature.melspectrogram(mono_sig,  # latest librosa deprecated logamplitude in favor of amplitude_to_db
    #    sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

    melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(mono_sig,
                                                                     sr=sr, n_mels=n_mels))[np.newaxis, :, :,
              np.newaxis]  # last newaxis is b/c tensorflow wants 'channels_last' order

    '''
    # librosa docs also include a perceptual CQT example:
    CQT = librosa.cqt(mono_sig, sr=sr, fmin=librosa.note_to_hz('A1'))
    freqs = librosa.cqt_frequencies(CQT.shape[0], fmin=librosa.note_to_hz('A1'))
    perceptual_CQT = librosa.perceptual_weighting(CQT**2, freqs, ref=np.max)
    melgram = perceptual_CQT[np.newaxis,np.newaxis,:,:]
    '''
    return melgram


def make_phase_gram(mono_sig, sr, n_bins=128):
    stft = librosa.stft(mono_sig)  # , n_fft = (2*n_bins)-1)
    magnitude, phase = librosa.magphase(stft)  # we don't need magnitude

    # resample the phase array to match n_bins
    phase = np.resize(phase, (n_bins, phase.shape[1]))[np.newaxis, :, :, np.newaxis]
    return phase


# turn multichannel audio as multiple melgram layers
def make_layered_melgram(signal, sr, mels=128, phase=False):
    if (signal.ndim == 1):  # given the way the preprocessing code is  now, this may not get called
        signal = np.reshape(signal, (1, signal.shape[0]))

    # get mel-spectrogram for each channel, and layer them into multi-dim array
    for channel in range(signal.shape[0]):
        melgram = make_melgram(signal[channel], sr, n_mels=mels)

        if (0 == channel):
            layers = melgram
        else:
            layers = np.append(layers, melgram,
                               axis=3)  # we keep axis=0 free for keras batches, axis=3 means 'channels_last'

        if (phase):
            phasegram = make_phase_gram(signal[channel], sr, n_bins=mels)
            layers = np.append(layers, phasegram, axis=3)
    return layers


def nearest_multiple(a, b):  # returns number smaller than a, which is the nearest multiple of b
    return int(a / b) * b


# can be used for test dataset as well
def build_dataset(path="Preproc/Train/", load_frac=1.0, batch_size=None, tile=False, max_per_class=0):
    '''

    :param path:
    :param load_frac:
    :param batch_size:
    :param tile:
    :param max_per_class:
    :return:
        X: 梅尔频谱图数据
        Y: onehot编码的label
    '''

    class_names = get_class_names(path=path)
    print("class_zie = ", len(class_names))
    print("class_names = ", class_names)
    print("max_per = ", max_per_class)
    nb_classes = len(class_names)

    total_files = get_total_files(class_names, path=path)
    total_load = int(total_files * load_frac)

    if max_per_class > 0:
        total_load = min(total_load, max_per_class * nb_classes)
    print("maxpercalss:", max_per_class)
    if (
            batch_size is not None):  # keras gets particular: dataset size must be mult. of batch_size 计算一共使用多少文件参与训练，batch_size整数倍
        total_load = nearest_multiple(total_load, batch_size)

    print("total files = ", total_files, ", going to load total_load = ", total_load)

    print("total files = ", total_files, ", going to load total_load = ", total_load)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    mel_dims = get_sample_dimensions(class_names, path=path)  # get dims of sample data file

    if (tile):
        ldims = list(mel_dims)
        ldims[3] = 3
        mel_dims = tuple(ldims)
    print(" melgram dimensions: ", mel_dims)
    X = np.zeros((total_load, mel_dims[1], mel_dims[2], mel_dims[3]))
    Y = np.zeros((total_load, nb_classes))
    paths = []

    load_count = 0
    for idx, classname in enumerate(class_names):
        print("")
        this_Y = np.array(encode_class(classname, class_names))
        this_Y = this_Y[np.newaxis, :]
        class_files = os.listdir(path + classname)
        shuffle(class_files)  # just to remove any special ordering
        n_files = len(class_files)
        n_load = int(n_files * load_frac)  # n_load is how many files of THIS CLASS are expected to be loaded
        if max_per_class > 0:
            n_load = min(n_load, max_per_class)
        printevery = 100

        file_list = class_files[0:n_load]

        for idx2, infilename in enumerate(file_list):  # Load files in a particular class
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery) or (idx2 + 1 == len(class_files)):
                print("\r Loading class ", idx + 1, "/", nb_classes, ": \'", classname,
                      "\', File ", idx2 + 1, "/", n_load, ": ", audio_path, "                  ",
                      sep="", end="")

            # auto-detect load method based on filename extension
            melgram = load_melgram(audio_path)
            if (tile) and (melgram.shape != mel_dims):
                melgram = np.tile(melgram, 3)
            elif (melgram.shape != mel_dims):
                print("\n\n    WARNING: Expecting spectrogram with dimensions mel_dims = ", mel_dims,
                      ", but got one with melgram.shape = ", melgram.shape)
                print("     The offending file is = ", audio_path)

            # usually it's the 2nd dimension of melgram.shape that is affected by audio file length
            use_len = min(X.shape[2], melgram.shape[2])
            X[load_count, :, 0:use_len] = melgram[:, :, 0:use_len]
            # X[load_count,:,:] = melgram
            Y[load_count, :] = this_Y
            paths.append(audio_path)
            load_count += 1
            if (load_count >= total_load):  # Abort loading files after last even multiple of batch size
                break

        if (load_count >= total_load):  # Second break needed to get out of loop over classes
            break

    print("")
    if (load_count != total_load):  # check to make sure we loaded everything we thought we would
        X = X[:load_count]
        Y = Y[:load_count]
        # raise Exception("Loaded "+str(load_count)+" files but was expecting "+str(total_load) )

    X, Y, paths = shuffle_XY_paths(X, Y, paths)  # mix up classes, & files within classes

    return X, Y, paths, class_names


def process_line(line):
    lst = re.split("[ \t]", line.strip())
    result = []
    for item in lst:
        if item == '':
            continue
        result.append(item)
    return result


def label_process(file='../label/label0507.txt', data_path='/data/voice/processed/', source_path='/data/voice/origin/'):
    top100Label = ['165',
                   '80',
                   '204',
                   '29',
                   '228',
                   '182',
                   '164',
                   '25',
                   '71',
                   '15',
                   '56',
                   '86',
                   '190',
                   '235',
                   '125',
                   '147',
                   '0',
                   '127',
                   '275',
                   '265',
                   '35',
                   '14',
                   '256',
                   '48',
                   '267',
                   '23',
                   '226',
                   '54',
                   '57',
                   '16',
                   '49',
                   '78',
                   '120',
                   '263',
                   '6',
                   '187',
                   '26',
                   '61',
                   '150',
                   '84',
                   '160',
                   '241',
                   '22',
                   '203',
                   '4',
                   '3',
                   '76',
                   '73',
                   '105',
                   '9',
                   '28',
                   '128',
                   '65',
                   '251',
                   '133',
                   '83',
                   '156',
                   '140',
                   '234',
                   '199',
                   '33',
                   '175',
                   '18',
                   '191',
                   '50',
                   '40',
                   '17',
                   '34',
                   '152',
                   '138',
                   '185',
                   '41',
                   '131',
                   '37',
                   '51',
                   '11',
                   '2',
                   '161',
                   '254',
                   '79',
                   '144',
                   '250',
                   '10',
                   '106',
                   '218',
                   '142',
                   '75',
                   '52',
                   '243',
                   '258',
                   '117',
                   '158',
                   '237',
                   '201',
                   '60',
                   '72',
                   '219',
                   '213',
                   '110',
                   '68',
                   '210',
                   '212',
                   '261',
                   '135',
                   '130',
                   '129',
                   '240',
                   '102',
                   '247',
                   '77',
                   '69',
                   '255',
                   '202',
                   '246',
                   '149',
                   '113',
                   '245',
                   '31',
                   '112',
                   '121',
                   '257',
                   '74',
                   '211',
                   '222',
                   '273',
                   '89',
                   '209',
                   '139',
                   '242',
                   '45',
                   '70',
                   '188',
                   '81',
                   '159',
                   '274',
                   '21',
                   '262',
                   '107',
                   '178',
                   '249',
                   '24',
                   '145',
                   '42',
                   '92',
                   '162',
                   '111',
                   '20',
                   '118',
                   '260',
                   '270',
                   '1',
                   '253',
                   '252',
                   '103',
                   '259',
                   '5',
                   '151',
                   '208',
                   '47',
                   '104',
                   '101',
                   '217',
                   '278',
                   '177',
                   '85',
                   '94',
                   '36',
                   '216',
                   '59',
                   '168',
                   '146',
                   '227',
                   '166',
                   '46',
                   '268']
    f = open(file).readlines()
    for line in tqdm(f):
        lst = process_line(line)
        if (len(lst) == 3):  # 有效数据 and lst[2] in top100Label
            target_path = data_path + lst[2]
            if (not os.path.exists(target_path)):
                os.makedirs(target_path)
            file_name = lst[0].split('/')[-1]
            shutil.copyfile(source_path + lst[0], os.path.join(target_path, file_name))


if __name__ == '__main__':
    label_process(file='/home/qjsun/work/VoiceClassification/data/label_copy/label_test.txt',
                  data_path='/data/voice/processed_test/', source_path='/data/voice/origin/')
