
import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_img


niftiFilesResampleDir = ""
niftiFilesDir = ""
niftiFilesNames = [s for s in os.listdir(niftiFilesDir) if s.endswith(".nii.gz")]
print("Files found: ", len(niftiFilesNames))


def loadAndResample(index, squareSize, numberOfSlices):

    niftiFile = os.path.join(niftiFilesDir, niftiFilesNames[index])
    theImage = nib.load(niftiFile)
    # print("Shape before resampling\t", theImage.shape)
    # print("zoom before resampling\t", theImage.header.get_zooms())

    target_shape = np.array((squareSize, squareSize, numberOfSlices))
    new_resolution = [(theImage.shape[0]*theImage.header.get_zooms()[0])/squareSize, (theImage.shape[1]*theImage.header.get_zooms()[1])/squareSize,
                      (theImage.shape[2]*theImage.header.get_zooms()[2])/numberOfSlices]
    print(new_resolution)

    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    new_affine[:3, 3] = theImage.affine[:3, 3]
    new_affine[3, 3] = 1.
    resampledNii = resample_img(theImage, target_affine=new_affine,target_shape=target_shape , interpolation='nearest')
    # print("Shape after resampling\t",resampledNii.shape)
    return resampledNii


for index, fileName in enumerate(niftiFilesNames):
    print("processing: ",index,fileName)
    resampledNii = loadAndResample(index, 256, 128)
    nib.save(resampledNii, os.path.join(niftiFilesResampleDir, niftiFilesNames[index]))


print("finished")