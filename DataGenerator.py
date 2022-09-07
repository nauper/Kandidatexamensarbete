
import numpy as np
import math

from nibabel.volumeutils import finite_range
from tensorflow.keras.utils import Sequence

import os
import nibabel as nib
import pandas as pd
import modelFunctions

split_folder_type = "-"
weighting_type = "-"

class DataGenerator(Sequence):

    def __init__(self, weighting_type, image_type, split_folder_type, batch_size, addDim, selLayers, startLayer, layerAmount):
        self.posCount = 0
        self.negCount = 0
        dataframe = pd.read_csv(modelFunctions.getPathToCSVFile(image_type))
        niftiFileNames = modelFunctions.getNiftiFileNames(split_folder_type, image_type, weighting_type)
        niftiFileLables = []  
        niftiImagesList = []
        count = 0     
        for niftiFileName in niftiFileNames:
            label = dataframe.loc[dataframe["Image ID"] == int(niftiFileName.split(".")[0])]
            label_value = label['Has Parkinson'].values[0]

            if label_value == 0:
                #niftiFileLables.append([0,1])
                niftiFileLables.append(0)
                self.negCount = self.negCount + 1
            else:
                #niftiFileLables.append([1,0])
                niftiFileLables.append(1)
                self.posCount = self.posCount + 1  

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
                #imageNpArray = imageNpArray[:, :, [72, 73, 76, 77, 78, 80, 81, 86]]
            niftiImagesList.append(imageNpArray)
            count += 1
            if(count % 100 == 0):
                print("File added: ", count, "/", len(niftiFileNames))
        
        self.x = np.array(niftiImagesList)
        if(addDim):
            self.x = self.x[..., np.newaxis]
            
        self.y = np.array(niftiFileLables)
        
        #self.x, self.y = niftiFileNames, niftiFileLables
        self.batch_size = batch_size
        self.split_folder_type = split_folder_type
        self.weighting_type = weighting_type
        self.image_type = image_type
        self.addDim = addDim
        self.selLayers = selLayers
        self.startLayer = startLayer
        self.layerAmount = layerAmount

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

