from data.datautils import load_audio, make_layered_melgram
from network.models import setup_model, MyCNN_Keras2
from data.process_data import get_canonical_shape
import numpy as np
from keras.models import Sequential, Model, load_model, save_model

def convert_one_file(file_path,resample, mono=False,
                     max_shape=(1, 432449),mels=96, phase=False):
    # infilename = class_files[file_index]
    audio_path = file_path
    # if not audio_path[-3:] == 'wav':
    #     return
    # if (0 == file_index % printevery) or (file_index+1 == len(class_files)):
    #     print("\r Processing class ",class_index+1,"/",nb_classes,": \'",classname,
    #         "\', File ",file_index+1,"/", n_load,": ",audio_path,"                             ",
    #         sep="",end="\r")
    sr = None
    if (resample is not None):
        sr = resample
    signal, sr = load_audio(audio_path, mono=mono, sr=sr)

    # Reshape / pad so all output files have same shape
    shape = get_canonical_shape(signal)     # either the signal shape or a leading one
    # print(shape)
    if(shape[1]>432449):
        raise Exception("音频长度过大")
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

    return np.expand_dims(layers, axis=1)

X = convert_one_file("./222.wav",resample=16000)
weights_file="/weights/weights_7250.hdf5"
# model, serial_model = setup_model(X_train, class_names, weights_file=weights_file, reshape_x=reshape_x,
#                                   drop_out_arg=drop_out_arg)
# serial_model = MyCNN_Keras2(X.shape, nb_classes=len(class_names), nb_layers=nb_layers, reshape_x=reshape_x,
#                                 drop_out_arg=drop_out_arg)
serial_model = MyCNN_Keras2(X.shape, nb_classes=163, nb_layers=4, reshape_x=52,
                                drop_out_arg=[0,0,0,0,0])
loaded_model = load_model(weights_file)  # strip any previous parallel part, to be added back in later
serial_model.set_weights(loaded_model.get_weights())  # assign weights based on checkpoint

pre = serial_model.predict(X)
print(pre.shape())

