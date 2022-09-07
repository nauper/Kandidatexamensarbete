
import numpy as np
import math

from nibabel.volumeutils import finite_range
from tensorflow.keras.utils import Sequence
import multiprocessing
import os
import nibabel as nib
import pandas as pd
import modelFunctions

def loadImages(procnum, dataframe, niftiFileNames, return_dict, weighting_type, image_type, split_folder_type, selLayers, startLayer, layerAmount):
    niftiFileLables = []  
    niftiImagesList = []
    negCount = 0
    posCount = 0
    for niftiFileName in niftiFileNames:
        label = dataframe.loc[dataframe["Image ID"] == int(niftiFileName.split(".")[0])]
        label_value = label['Has Parkinson'].values[0]

        if label_value == 0:
            #niftiFileLables.append([0,1])
            niftiFileLables.append(0)
            negCount = negCount + 1
        else:
            #niftiFileLables.append([1,0])
            niftiFileLables.append(1)
            posCount = posCount + 1  

        niftiFile = os.path.join(modelFunctions.getPathToNiftiFiles(weighting_type, image_type, split_folder_type), niftiFileName)
        theImage = nib.load(niftiFile)
        imageNpArray = theImage.get_fdata()
        if(image_type == "mri" or image_type == "mriPre"):            
            reduce_d = modelFunctions.get_cut_amount()
            if(selLayers):
                imageNpArray = imageNpArray[reduce_d:-reduce_d, reduce_d:-reduce_d, :]
            else:
                imageNpArray = imageNpArray[reduce_d:-reduce_d, reduce_d:-reduce_d, reduce_d:-reduce_d]   
            min, max = finite_range(imageNpArray)
            if max == 0:
                print("zero max in: "+niftiFileName)
            #print("max min: ", max, " ", min)
            imageNpArray = np.float16((imageNpArray / max) * 255)
        if(selLayers):
            imageNpArray = imageNpArray[:, :, startLayer:startLayer+layerAmount]
        niftiImagesList.append(imageNpArray)
        
       
    print("Done: ", len(niftiFileNames))
    return_dict[procnum] = [niftiImagesList, niftiFileLables, posCount, negCount]

class DataGenerator(Sequence):

    def __init__(self, weighting_type, image_type, split_folder_type, batch_size, addDim, selLayers, startLayer, layerAmount):
        self.posCount = 0
        self.negCount = 0

        self.batch_size = batch_size
        self.split_folder_type = split_folder_type
        self.weighting_type = weighting_type
        self.image_type = image_type
        self.addDim = addDim
        self.selLayers = selLayers
        self.startLayer = startLayer
        self.layerAmount = layerAmount
        
      
        
    def start(self, weighting_type, image_type, split_folder_type, batch_size, addDim, selLayers, startLayer, layerAmount):        
        dataframe = pd.read_csv(modelFunctions.getPathToCSVFile(image_type))
        niftiFileNames = modelFunctions.getNiftiFileNames(split_folder_type, image_type, weighting_type)
        niftiFileLables = []  
        niftiImagesList = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        subProcesses = 16
        nifitiSplit = np.array_split(niftiFileNames, subProcesses)
        for i in range(subProcesses):
            p = multiprocessing.Process(target=loadImages, args=(i, dataframe, nifitiSplit[i], return_dict, weighting_type, image_type, split_folder_type, selLayers, startLayer, layerAmount))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        for i in range(subProcesses):
            niftiImagesList + return_dict[i][0]
            niftiImagesList + return_dict[i][1]
            self.posCount += return_dict[i][2]
            self.negCount += return_dict[i][3]
            
        self.x = np.array(niftiImagesList)
        if(addDim):
            self.x = self.x[..., np.newaxis]
            
        self.y = np.array(niftiFileLables)  
        

    def __len__(self):
        lenthOfDataset = math.ceil(len(self.x) / self.batch_size)
        #print("the length of dataset is: {}".format(lenthOfDataset), self.split_folder_type)
        return lenthOfDataset

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        #print("\n batch {} shape x: {}   shape y: {}  \n".format(idx, xNpArray.shape,yNpArray.shape))
        return batch_x, batch_y

