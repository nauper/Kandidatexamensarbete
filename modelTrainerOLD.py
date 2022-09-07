# this class is for training the model
# We want to input a 3d-brain-image and get 0 or 1
# as a result (has not parkinson or has parkinson)


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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





#Show device placement
#tf.debugging.set_log_device_placement(True)

#Run on cpu
#tf.config.set_visible_devices([], 'GPU')


batch_size = 6
#spect or mri
image_type = "spect"
# use lowercase: pd, t1, t2
preferred_weighting = "pd"

testSpan = 1
epochs = 100
workers = 4
max_queue_size = 20
verbose = 2
learning_rate = 0.0001
# -----------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------
input_resized_shapes = []
if(image_type == "mri"):    
    input_original_shapes = [182, 218, 182]
    input_x_shape = input_original_shapes[0] - 2 * modelFunctions.get_cut_amount()
    input_y_shape = input_original_shapes[1] - 2 * modelFunctions.get_cut_amount()
    input_z_shape = input_original_shapes[2] - 2 * modelFunctions.get_cut_amount()
    input_resized_shapes = [input_x_shape, input_y_shape, input_z_shape]
elif(image_type == "spect"): 
    input_resized_shapes = [91, 109, 91]  # SPECT images dimension






# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=modelFunctions.createSavingLocation(preferred_weighting, image_type),
                                                 save_weights_only=True,
                                                 verbose=0)
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
# MODEL
# -----------------------------------------------------------------

model = modelFunctions.get_model_6(input_resized_shapes)

# -----------------------------------------------------------------
# MODEL COMPILE
# -----------------------------------------------------------------

#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=METRICS)

#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses.binary_crossentropy, metrics=METRICS)
#model.summary()

# -----------------------------------------------------------------
# RELOAD WEIGHTS
# -----------------------------------------------------------------

# model.load_weights(checkpoint_path)


# -----------------------------------------------------------------
# GENERATORS
# -----------------------------------------------------------------
train_generator = DataGenerator(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="train", batch_size=batch_size)
validation_generator = DataGenerator(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="valid", batch_size=batch_size)
test_generator = DataGenerator(weighting_type=preferred_weighting, image_type=image_type, split_folder_type="test", batch_size=batch_size)

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


# #modelFunctions.plot_metrics(history)
# -------------- display model summary as SVG for report ------------------
#SVG(vis_utils.model_to_dot(model, show_shapes=True).create_svg())


def train(testSpan, epochs, workers, max_queue_size, verbose, learning_rate):    
    tic = time.perf_counter()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses.binary_crossentropy, metrics=METRICS)
    model.summary()
    history = {}
    results = []

    for x in range(0, epochs, testSpan):
        tic2 = time.perf_counter()
        historyTemp = model.fit(x=train_generator, validation_data=validation_generator, epochs=testSpan, verbose=verbose,  callbacks=[cp_callback],max_queue_size=max_queue_size, workers=workers, class_weight=class_weight)
        history = modelFunctions.mergeDicts(history, historyTemp.history)
        resultsTemp = model.evaluate(x=test_generator, verbose=verbose)
        results = modelFunctions.mergeResults(results, resultsTemp)
        toc2 = time.perf_counter()
        totTime2 = toc2 - tic2
        print("Epoch: ", x+testSpan, " / ", epochs, "   Time: ", totTime2)

    toc = time.perf_counter()
    totTime = toc - tic

    print(f"Epochs: {epochs} Learning Rate: {learning_rate} Workers: {workers} MQS: {max_queue_size} ", f" {totTime:0.4f} sec", f"     Loss: {results[0]}, TP: {results[1]}, FP: {results[2]}, TN: {results[3]}, FN: {results[4]}, Accuracy: {results[5]}, Recall: {results[6]}, Precision: {results[7]}")
    modelFunctions.writeHistoryToFile(history, results, preferred_weighting, image_type, epochs, learning_rate, batch_size, totTime, workers, max_queue_size)

#train(test every # epoch, epochs, workers, max_queue_size, verbose, callbacks):
train(1, 100, 4, 20, 2, 0.0001)

#workNr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# mqs = [1, 5, 10, 25, 50, 150, 400]
# learningRates = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
# for x in learningRates:
#     train(100, 4, 10, 2, cp_callback, x)


# for y in workNr:
#     for x in mqs:
#       train(y, x, 1)