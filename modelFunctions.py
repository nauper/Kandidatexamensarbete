import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Dense, Conv3D, MaxPool3D, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Activation, MaxPooling2D
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Input
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling3D

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# reduce_dimentions_by_Pixels: cut this many pixels from every the sides of the brain because they might not contain any value anyway
cut_sides_by_this_amount = 10
def writeHistoryToFile(dir_to_create, fileName, history, results, batch_size, image_type, weighting_type, modelName, addDim, selectLayers, startLayer, layerAmount, testSpan, epochs, workers, max_queue_size, learning_rate, totTime, pos, neg, total):
    try:
        os.makedirs(dir_to_create)
    except OSError:
        temp = 1
        
    else:
        print("Successfully created the directory %s " % dir_to_create)

    
    df1 = pd.DataFrame(history).T
    df2 = pd.DataFrame(results).T
    df2.columns=["Loss:", "TP:", "FP:", "TN:", "FN:", "Accuracy:", "Recall:", "Precision:"]
    try:
        with pd.ExcelWriter(os.path.join(dir_to_create, fileName+".xlsx")) as writer:
            df1.to_excel(writer, sheet_name='History')
            df2.to_excel(writer, sheet_name='Results')
    except:
        with pd.ExcelWriter(os.path.join(dir_to_create, fileName+"2.xlsx")) as writer:
            df1.to_excel(writer, sheet_name='History')
            df2.to_excel(writer, sheet_name='Results')


    f = open(os.path.join(dir_to_create, "info.txt"), "w")
    f.write(f"Image Type: {image_type} \nWeighting Type: {weighting_type} \nModel: {modelName}\nPositve Images: {pos} \nNegative Images: {neg} \nTotalt: {total} \n\nEpochs: {epochs} \nLearning Rate: {learning_rate} \nBatch Size: {batch_size} \nTime: {totTime:0.4f} sec \nAdded dimension: {addDim} \nSelected Layers: {selectLayers} \nStart Layer: {startLayer} \nNumber of Layers: {layerAmount} \nTest every: {testSpan} \nWorkers: {workers} \nMQS: {max_queue_size} \n\nLoss: {results[0]} \nTP: {results[1]} \nFP: {results[2]} \nTN: {results[3]} \nFN: {results[4]} \nAccuracy: {results[5]} \nRecall: {results[6]} \nPrecision: {results[7]}")
    f.close()
def writeHistoryToFileAppend(dir_to_create, fileName, sheetID, history, results, batch_size, image_type, weighting_type, modelName, addDim, selectLayers, startLayer, layerAmount, testSpan, epochs, workers, max_queue_size, learning_rate, totTime, pos, neg, total):
    try:
        os.makedirs(dir_to_create)
    except OSError:
        temp = 1
        
    else:
        print("Successfully created the directory %s " % dir_to_create)

    
    df1 = pd.DataFrame(history).T
    df2 = pd.DataFrame(results).T
    df2.columns=["Loss:", "TP:", "FP:", "TN:", "FN:", "Accuracy:", "Recall:", "Precision:"]
    try:
        with pd.ExcelWriter(os.path.join(dir_to_create, fileName+".xlsx"), engine="openpyxl", mode='a') as writer:
            df1.to_excel(writer, sheet_name="History"+str(sheetID))
            df2.to_excel(writer, sheet_name="Results"+str(sheetID))
    except FileNotFoundError:
        with pd.ExcelWriter(os.path.join(dir_to_create, fileName+".xlsx"), engine="openpyxl") as writer:
            df1.to_excel(writer, sheet_name="History"+str(sheetID))
            df2.to_excel(writer, sheet_name="Results"+str(sheetID))
    except PermissionError:
        with pd.ExcelWriter(os.path.join(dir_to_create, fileName+ datetime.now().strftime("%H-%M")+".xlsx")) as writer:
            df1.to_excel(writer, sheet_name="History"+str(sheetID))
            df2.to_excel(writer, sheet_name="Results"+str(sheetID))


    f = open(os.path.join(dir_to_create, "info"+str(sheetID)+".txt"), "w")
    f.write(f"Image Type: {image_type} \nWeighting Type: {weighting_type} \nModel: {modelName}\nPositve Images: {pos} \nNegative Images: {neg} \nTotalt: {total} \n\nEpochs: {epochs} \nLearning Rate: {learning_rate} \nBatch Size: {batch_size} \nTime: {totTime:0.4f} sec \nAdded dimension: {addDim} \nSelected Layers: {selectLayers} \nStart Layer: {startLayer} \nNumber of Layers: {layerAmount} \nTest every: {testSpan} \nWorkers: {workers} \nMQS: {max_queue_size} \n\nLoss: {results[0]} \nTP: {results[1]} \nFP: {results[2]} \nTN: {results[3]} \nFN: {results[4]} \nAccuracy: {results[5]} \nRecall: {results[6]} \nPrecision: {results[7]}")
    f.close()    
def mergeDicts(dict1, dict2):
    if(dict1):
        for x in dict2:
            dict1[x] += dict2.get(x)
    else:
        dict1 = dict2    
    return dict1       
def mergeResults(res1, res2):
    count = len(res1)
    if(count > 0):
        for x in range(count):
            res1[x].append(res2[x])   
    else:
        for x in range(len(res2)):
            res1.append([res2[x]]) 
    return res1  
def getPathToNiftiFiles(weight_type, image_type, split_folder_type):
    # group_type can be train, test or valid
    return os.path.join(os.getcwd(), image_type, weight_type, split_folder_type)

def getPathToCSVFile(image_type):
    return os.path.join(os.getcwd(), image_type, "csv") + "/combined.csv"


def createSavingLocation(saveDirPath, fileN):
    # detect the current working directory and print it
    dir_to_create = os.path.join(saveDirPath, "saved_weights")
    try:
        os.makedirs(dir_to_create)
    except OSError:
        print("Creation of the directory %s failed" % dir_to_create)
    else:
        print("Successfully created the directory %s " % dir_to_create)

    return dir_to_create + fileN

def getNiftiFileNames(split_folder_type, image_type, weighting_type):
    niftiFileNamesFound = [s for s in os.listdir(getPathToNiftiFiles(weighting_type, image_type, split_folder_type)) if s.endswith(".nii")]
    if(len(niftiFileNamesFound) == 0):
        niftiFileNamesFound = [s for s in os.listdir(getPathToNiftiFiles(weighting_type, image_type, split_folder_type)) if s.endswith(".nii.gz")]
    print(f"{image_type} {weighting_type} {split_folder_type} files found: {len(niftiFileNamesFound)}")

    return niftiFileNamesFound




# reduce_dimentions_by_Pixels: cut this many pixels from every the sides of the brain because they might not contain any value anyway
def get_cut_amount():
    return cut_sides_by_this_amount


def get_model_1(input_resized_shapes):
    model = Sequential()
    # block 2
    model.add(Conv2D(filters=8, kernel_size=3, padding="valid", activation='relu',
                     input_shape=(input_resized_shapes[0],input_resized_shapes[1],input_resized_shapes[2])))
    model.add(MaxPool2D(3))
    # block 3
    model.add(Conv2D(filters=16, kernel_size=3, padding="valid", activation='relu'))
    model.add(MaxPool2D(3))
    # block 3
    model.add(Conv2D(filters=32, kernel_size=3, padding="valid", activation='relu'))
    model.add(MaxPool2D(3))
    # block 4
    model.add(BatchNormalization())
    model.add(Flatten())
    # add elements together
    # model.add(GlobalAvgPool3D())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model



# VGG-16 like network
def get_model_2(input_resized_shapes):
    # In vgg-16 all conv layer have 3*3*3 kernel with stride of 1 and same padding
    # all max-poolings are 2*2*2 with stride of 2
    kernel = 3
    stride = 1
    padding = "same"
    max_pooling_size = 2
    pool_stride = 2

    model = Sequential()
    # block 1
    model.add(Conv2D(filters=64, kernel_size=kernel, padding=padding, strides=stride, activation='relu',
                     input_shape=(input_resized_shapes[0],input_resized_shapes[1],input_resized_shapes[2])))
    model.add(Conv2D(filters=64, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))

    # block 2
    model.add(MaxPool2D(pool_size=max_pooling_size, strides=pool_stride))

    # block 3
    model.add(Conv2D(filters=128, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))

    # block 4
    model.add(MaxPool2D(pool_size=max_pooling_size, strides=pool_stride))

    # block 5
    model.add(Conv2D(filters=256, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))

    # block 6
    model.add(MaxPool2D(pool_size=max_pooling_size, strides=pool_stride))

    # block 7
    model.add(Conv2D(filters=512, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))

    # block 8
    model.add(MaxPool2D(pool_size=max_pooling_size, strides=pool_stride))

    # block 9
    model.add(Conv2D(filters=512, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=kernel, padding=padding, strides=stride, activation='relu'))

    # block 10
    model.add(MaxPool2D(pool_size=max_pooling_size, strides=pool_stride))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


def get_model_3(input_resized_shapes):
    model = Sequential([Conv2D(128, (5, 5),  padding='same', kernel_regularizer=l2(0.001), activation='relu', input_shape=(input_resized_shapes[0],input_resized_shapes[1],input_resized_shapes[2])),
                        Conv2D(64, (3, 3),  padding='same', kernel_regularizer=l2(0.001), activation='relu'),
                        MaxPool2D(pool_size=(2, 2)),
                        BatchNormalization(),
                        Flatten(),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid' )
                       ])
    return model
def get_model_4(input_resized_shapes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(input_resized_shapes[0],input_resized_shapes[1],input_resized_shapes[2])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
def get_model_5(input_resized_shapes):
    model=Sequential()
    model.add(Conv2D(filters=16, kernel_size=(1, 32),activation='relu', input_shape=(input_resized_shapes[0],input_resized_shapes[1],input_resized_shapes[2]),padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Conv2D(filters=32, kernel_size=(1, 16), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 6)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
def get_model_6(input_resized_shapes):
    """Build a 3D convolutional neural network model."""
    model = Sequential()
    model.add(Input(shape=(input_resized_shapes[0], input_resized_shapes[1], input_resized_shapes[2], 1)))

    model.add(Conv3D(filters=64, kernel_size=3, activation="relu", padding='same'))
    model.add(MaxPool3D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=64, kernel_size=3, activation="relu", padding='same'))
    model.add(MaxPool3D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=128, kernel_size=3, activation="relu", padding='same'))
    model.add(MaxPool3D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=256, kernel_size=3, activation="relu", padding='same'))
    model.add(MaxPool3D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling3D())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(units=1, activation="sigmoid"))
    return model
def get_model_7(input_resized_shapes):
    """Build a 3D convolutional neural network model."""
    model = Sequential()
    model.add(Input(shape=(input_resized_shapes[0], input_resized_shapes[1], input_resized_shapes[2], 1)))

    model.add(Conv3D(filters=8, kernel_size=3, activation="relu"))
    model.add(MaxPool3D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=16, kernel_size=3, activation="relu"))
    model.add(MaxPool3D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=32, kernel_size=3, activation="relu"))
    model.add(MaxPool3D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPool3D(pool_size=2))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling3D())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(units=1, activation="sigmoid"))
    return model
