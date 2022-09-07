# this class is for training the model
# We want to input a 3d-brain-image and get 0 or 1
# as a result (has not parkinson or has parkinson)
import os
import multiprocessing as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Make numpy values easier to read.
# np.set_printoptions(precision=3, suppress=True)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, MaxPool3D, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras import optimizers, losses
from DataGenerator import DataGenerator
from tensorflow.python.keras.utils import vis_utils
from IPython.display import SVG
import modelFunctions
import time
from datetime import datetime
#Show device placement
#tf.debugging.set_log_device_placement(True)

#Run on cpu
#tf.config.set_visible_devices([], 'GPU')

batch_size = 64
#spect or mri
image_type = "mri"
# use lowercase: pd, t1, t2
preferred_weighting = "t1"
modelFunc = modelFunctions.get_model_5
addDim = False

saveEvery = 3000
saveWeights = False
#74 good
selectLayers = True
startLayer = 82
layerAmount = 7


testSpan = 1
epochs = 2000
workers = 8
max_queue_size = 32
verbose = 0
learning_rate = 0.00001



def train(batch_size, workers, max_queue_size, learning_rate, sheetID, saveDirPath, startLayer, layerAmount):
    print(f"New Start - Start Layer: {startLayer} Layer Amount: {layerAmount}")
    tic = time.perf_counter() 
    input_resized_shapes = []
    if(image_type == "mri"):    
        input_original_shapes = [182, 218, 182]
        input_x_shape = input_original_shapes[0] - 2 * modelFunctions.get_cut_amount()
        input_y_shape = input_original_shapes[1] - 2 * modelFunctions.get_cut_amount()
        input_z_shape = input_original_shapes[2] - 2 * modelFunctions.get_cut_amount()
        if(selectLayers):
            input_resized_shapes = [input_x_shape, input_y_shape, layerAmount]
        else:
            input_resized_shapes = [input_x_shape, input_y_shape, input_z_shape]
    elif(image_type == "mriPre"):
        input_original_shapes = [176, 240, 256]
        input_x_shape = input_original_shapes[0] - 2 * modelFunctions.get_cut_amount()
        input_y_shape = input_original_shapes[1] - 2 * modelFunctions.get_cut_amount()
        input_z_shape = input_original_shapes[2] - 2 * modelFunctions.get_cut_amount()
        if(selectLayers):
            input_resized_shapes = [input_x_shape, input_y_shape, layerAmount]
        else:
            input_resized_shapes = [input_x_shape, input_y_shape, input_z_shape]
    elif(image_type == "spect"): 
        if(selectLayers):
            input_resized_shapes = [91, 109, layerAmount]
        else:
            input_resized_shapes = [91, 109, 91]  # SPECT images dimension   
        
    model = modelFunc(input_resized_shapes)
        # Create a callback that saves the model's weights
   
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=modelFunctions.createSavingLocation(saveDirPath, "/modelTrainer.ckpt"), save_weights_only=True, verbose=0)
    


    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),  #True negatives and true positives are samples that were correctly classified
        tf.keras.metrics.FalseNegatives(name='fn'), #False negatives and false positives are samples that were incorrectly classified
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='recall (Sensitivity)'),
        tf.keras.metrics.Precision(name='precision (Specificity)'),
        # tf.keras.metrics.AUC(name='auc'),
    ]
    # -----------------------------------------------------------------
    # GENERATORS
    # -----------------------------------------------------------------
    ticDataLoad = time.perf_counter()
    train_generator = DataGenerator(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="train", batch_size=batch_size, addDim=addDim, selLayers=selectLayers, startLayer=startLayer, layerAmount=layerAmount)
    validation_generator = DataGenerator(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="valid", batch_size=batch_size, addDim=addDim, selLayers=selectLayers, startLayer=startLayer, layerAmount=layerAmount)
    test_generator = DataGenerator(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="test", batch_size=batch_size, addDim=addDim, selLayers=selectLayers, startLayer=startLayer, layerAmount=layerAmount)
    
    # train_generator.start(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="train", batch_size=batch_size, addDim=addDim, selLayers=selectLayers, startLayer=startLayer, layerAmount=layerAmount)
    # validation_generator.start(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="valid", batch_size=batch_size, addDim=addDim, selLayers=selectLayers, startLayer=startLayer, layerAmount=layerAmount)
    # test_generator.start(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="test", batch_size=batch_size, addDim=addDim, selLayers=selectLayers, startLayer=startLayer, layerAmount=layerAmount)
    
    print(f"Loading Time: {time.perf_counter()-ticDataLoad:0.2f}s")

    # -----------------------------------------------------------------
    # TRAIN
    # -----------------------------------------------------------------
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.

    pos = train_generator.posCount
    neg = train_generator.negCount
    total = pos + neg
    weight_for_0 = pos / total   #control
    weight_for_1 = neg / total   #pd
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f'Pos: {pos} Neg: {neg} Totalt: {total}')
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))


    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses.binary_crossentropy, metrics=METRICS)
    #model.summary()
    history = {}
    results = []

    for x in range(0, epochs, testSpan):
        tic2 = time.perf_counter()

        ticTrain = time.perf_counter()
        if(saveWeights):
            historyTemp = model.fit(x=train_generator, validation_data=validation_generator, epochs=testSpan, verbose=verbose,  callbacks=[cp_callback],max_queue_size=max_queue_size, workers=workers, class_weight=class_weight)
        else:
            historyTemp = model.fit(x=train_generator, validation_data=validation_generator, epochs=testSpan, verbose=verbose, max_queue_size=max_queue_size, workers=workers, class_weight=class_weight)

        history = modelFunctions.mergeDicts(history, historyTemp.history)
        totTrain = time.perf_counter()-ticTrain

        ticEvaluate = time.perf_counter()
        resultsTemp = model.evaluate(x=test_generator, verbose=verbose)
        results = modelFunctions.mergeResults(results, resultsTemp)  
        totEvaluate = time.perf_counter() - ticEvaluate

        lastTen = results[5][-10:]
        movingAverage = sum(lastTen)/len(lastTen)
        
        if(x % saveEvery == 0):
            totTime = time.perf_counter() - tic
            ticSave = time.perf_counter()
            modelFunctions.writeHistoryToFileAppend(saveDirPath, "history", sheetID,history, results, batch_size, image_type, preferred_weighting, modelFunc.__name__, addDim, selectLayers, startLayer, layerAmount, testSpan, epochs, workers, max_queue_size, learning_rate, totTime, pos, neg, total)
            print(f"Saving Time: {time.perf_counter()-ticSave:0.2f}s")
        totTime2 = time.perf_counter()-tic2
        print(f"Epoch: {x+testSpan}/{epochs}\tTime (train, evaluate): {totTime2:0.2f}s ({totTrain:0.2f}s, {totEvaluate:0.2f}s)   Loss: {historyTemp.history.get('loss')[0]:0.4f}   Accuracy: {historyTemp.history.get('accuracy')[0]:0.4f}   Val Accuracy: {historyTemp.history.get('val_accuracy')[0]:0.4f}   Test Accuracy: {resultsTemp[5]:0.4f}   Moving Avergage: {movingAverage:0.4f}")



    totTime = time.perf_counter() - tic
    ticSave = time.perf_counter()
    modelFunctions.writeHistoryToFileAppend(saveDirPath, "historyFinal", sheetID,history, results, batch_size, image_type, preferred_weighting, modelFunc.__name__, addDim, selectLayers, startLayer, layerAmount, testSpan, epochs, workers, max_queue_size, learning_rate, totTime, pos, neg, total)
    print(f"Saving Final Time: {time.perf_counter()-ticSave:0.2f}s")

    #print(f"Epochs: {epochs} Learning Rate: {learning_rate} Workers: {workers} MQS: {max_queue_size} ", f" {totTime:0.4f} sec", f"     Loss: {results[0]}, TP: {results[1]}, FP: {results[2]}, TN: {results[3]}, FN: {results[4]}, Accuracy: {results[5]}, Recall: {results[6]}, Precision: {results[7]}")
    #modelFunctions.writeHistoryToFile(saveDirPath, history, results, batch_size, image_type, preferred_weighting, modelFunc.__name__, addDim, selectLayers, startLayer, layerAmount, testSpan, epochs, workers, max_queue_size, learning_rate, totTime, pos, neg, total)
    #modelFunctions.writeHistoryToFile(history, results, preferred_weighting, image_type, epochs, learning_rate, batch_size, totTime, workers, max_queue_size, )

    ## GARBAGE COLLECT
    del train_generator
    del test_generator
    del validation_generator
    del history
    del results
    del historyTemp
    del resultsTemp

if __name__ == '__main__':
    current_dir_path = os.getcwd()
    saveDirPath = os.path.join(current_dir_path, image_type, preferred_weighting, "saved_history",datetime.now().strftime("%Y-%m-%d-kl-%H-%M"))

    #proc = mp.Process(target=train, args=(batch_size, workers, max_queue_size, learning_rate, "1", saveDirPath, startLayer, layerAmount))
    #proc.start()
    #proc.join()

    #for x in range(9, 11, 2):
    ticStart = time.perf_counter()
    layerCounts = [1, 41, 81, 121]
    layerAmounts = [40, 40, 40, 42]
    for y in range(1, 6):
        for x in range(0, 4):        
            ticRunStart = time.perf_counter()
            startLayer = layerCounts[x]
            layerAmount = layerAmounts[x]
            print(f"Start Layer: {startLayer} Layer Amount: {layerAmount} Run: {y}" )
            proc = mp.Process(target=train, args=(batch_size, workers, max_queue_size, learning_rate, "L"+str(startLayer)+"R"+str(y), saveDirPath, startLayer, layerAmount))
            proc.start()
            proc.join()

            print(f"Totalt runtime: {time.perf_counter()-ticRunStart:0.2f}s Start Layer: {startLayer} Layer Amount: {layerAmount} Run: {y}")
    
    
    
    print(f"Totalt program runtime: {time.perf_counter()-ticStart:0.2f}s")

# if __name__ == '__main__':
#     startLayer = 65
#     proc = mp.Process(target=train, args=(batch_size, workers, max_queue_size, learning_rate, "L"+str(startLayer)+"R1"))
#     proc2 = mp.Process(target=train, args=(batch_size, workers, max_queue_size, learning_rate, "L"+str(startLayer)+"R2"))
#     proc3 = mp.Process(target=train, args=(batch_size, workers, max_queue_size, learning_rate, "L"+str(startLayer)+"R3"))
#     proc4 = mp.Process(target=train, args=(batch_size, workers, max_queue_size, learning_rate, "L"+str(startLayer)+"R4"))
#     proc5 = mp.Process(target=train, args=(batch_size, workers, max_queue_size, learning_rate, "L"+str(startLayer)+"R5"))

#     proc.start()
#     proc2.start()
    

# train(batch_size, workers, max_queue_size, learning_rate, 1)




#workNr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# mqs = [1, 5, 10, 25, 50, 150, 400]
# learningRates = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
# for x in learningRates:
#     train(100, 4, 10, 2, cp_callback, x)


# for y in workNr:
#     for x in mqs:
#       train(y, x, 1)