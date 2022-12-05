from collections.abc import MutableMapping

import tensorflow as tf
import numpy as np
import keras
from os import listdir
from os.path import isfile, join
import re
import pandas as pd

import argparse

from code.augmentation import run_augmentation

dirname = "../../pretrained_models_ucr/UCR_TS_Archive_2015/"
onlyfiles = [f for f in listdir(dirname)]

x_train_list = [np.load("data/TL/x_train" + str(i) + ".npy") for i in range(1,11)]
x_test_list = [np.load("data/TL/x_test" + str(i) + ".npy") for i in range(1,11)]
y_train_list = [np.load("data/TL/y_train" + str(i) + ".npy") for i in range(1,11)]
y_test_list = [np.load("data/TL/y_test" + str(i) + ".npy") for i in range(1,11)]

input_shape = x_train_list[0].shape[1:]

# aug ratios
augmentation_ratios = [0,2,4,8,12]

### transfer learn function
def transfer_learn(input_shape, prtrmod):
    
    resh = False
    if prtrmod.layers[0].input_shape[0][2] == 1:
        input_shape = (input_shape[0]*input_shape[1], 1)
        resh = True
    inp_layer = keras.layers.Input(shape=input_shape)
    prtrlay = prtrmod.layers[1:-1]
    for i in range(len(prtrlay)):
        prtrlay[i].trainable = False
    mod = keras.Sequential([inp_layer] + 
                           prtrlay + 
                           [keras.layers.Dense(units=5, activation="softmax")])
    
    mod.compile(
        optimizer = keras.optimizers.Adam(),
        loss = "categorical_crossentropy",
        metrics = "accuracy"
    )
    
    return(mod, resh)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs augmentation model.')
  
    # Augmentation
    # parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=True, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=True, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=True, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=True, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=True, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    args = parser.parse_args()
    
    for data_ind in reversed(range(len(x_train_list))):
	
        print("Data set " + str(data_ind) + "/10 \n")
        
        x_train_org = x_train_list[data_ind]
        x_test = x_test_list[data_ind]
        y_train_org = keras.utils.to_categorical(y_train_list[data_ind]-1, 5)
        y_test = y_test_list[data_ind]
            
        for augrat in augmentation_ratios:
            
            args.augmentation_ratio = augrat
            args.dataset = ""
            
            x_train, y_train, _ = run_augmentation(x_train_org, y_train_org, args)
    
            for flname in onlyfiles:
    
                # print("Check if results exist...\n")    
    
                if flname + "_aug_x" + str(augrat) + "_fold_" + str(data_ind) + ".csv" in listdir("output/TL_AUG"):
                    continue
    
                # if sum(mt)==10:
                  #  continue
        
                print("Transfer-learning on " + flname + "\n")
    
                fns = listdir(dirname + flname)
                fn = [fn for fn in fns  if re.match(r"best_model\.hdf5$", fn) is not None][0]
                mod = keras.models.load_model(dirname + flname + "/" + fn)
                mod, resh = transfer_learn(input_shape, mod)
                
                if resh:
                    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2],1))
                    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2],1))

                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        verbose=0
                    )
                ]
        
                print("Learning network head...\n")
        
                history = mod.fit(
                    x_train,
                    y_train,
                    batch_size=64,
                    epochs=50,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose = False
                )
        
                print("Predicting...\n")

                y_hat = mod.predict(x_test, verbose = False)
        
                res = pd.DataFrame(np.c_[mod.predict(x_test), y_test], 
                                   index=['Row'+str(i) for i in range(1, len(y_test_list[data_ind])+1)])
                res.columns = ['1', '2', '3', '4', '5', 'truth']
        
                res.to_csv("output/TL_AUG/" + flname + "_aug_x" + str(augrat) + "_fold_" + str(data_ind) + ".csv")
