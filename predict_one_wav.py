from datautils import load_audio, make_layered_melgram
from process_data import get_canonical_shape
import numpy as np

def convert_one_file(printevery, class_index, class_files, nb_classes, classname, n_load, dirname, resample, mono,
        already_split, nosplit, n_train, outpath, subdir, max_shape, clean, out_format, mels, phase, file_index):
    infilename = class_files[file_index]
    audio_path = dirname + '/' + infilename
    # if not audio_path[-3:] == 'wav':
    #     return
    if (0 == file_index % printevery) or (file_index+1 == len(class_files)):
        print("\r Processing class ",class_index+1,"/",nb_classes,": \'",classname,
            "\', File ",file_index+1,"/", n_load,": ",audio_path,"                             ",
            sep="",end="\r")
    sr = None
    if (resample is not None):
        sr = resample
    signal, sr = load_audio(audio_path, mono=mono, sr=sr)

    # Reshape / pad so all output files have same shape
    shape = get_canonical_shape(signal)     # either the signal shape or a leading one
    # print(shape)
    if(shape[1]>432449):
        return
    if (shape != signal.shape):             # this only evals to true for mono
        signal = np.reshape(signal, shape)
        #print("...reshaped mono so new shape = ",signal.shape, end="")
    #print(",  max_shape = ",max_shape,end="")
    padded_signal = np.zeros(max_shape)     # (previously found max_shape) allocate a long signal of zeros
    use_shape = list(max_shape[:])
    use_shape[0] = min( shape[0], max_shape[0] )
    use_shape[1] = min( shape[1], max_shape[1] )
    #print(",  use_shape = ",use_shape)
    padded_signal[:use_shape[0], :use_shape[1]] = signal[:use_shape[0], :use_shape[1]]

    layers = make_layered_melgram(padded_signal, sr, mels=mels, phase=phase)

    if not already_split and (not nosplit):
        if (file_index >= n_train):
            outsub = "Test/"
        else:
            outsub = "Train/"
    elif nosplit:
        outsub = ""
    else:
        outsub = subdir

    outfile = outpath + outsub + classname + '/' + infilename+'.'+out_format
    save_melgram(outfile, layers, out_format=out_format)
    return