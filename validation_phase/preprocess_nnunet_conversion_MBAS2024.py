# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:52:09 2024

@author: aq22
"""

#%% Left and right atrium segmentation pytorch dataset
import os
from glob import glob
import nibabel as nib
path='C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/MICCAI_MBAS_2024/MBAS_Dataset/Training'
imges_path=glob(os.path.join(path,'*','*_gt.nii.gz'))
labels_path=glob(os.path.join(path,'*','*_label.nii.gz'))
pathsaveim='C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/MICCAI_MBAS_2024/MBAS_Dataset/nn_dataset/preprocessed/imagesTr'
pathsavegt='C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/MICCAI_MBAS_2024/MBAS_Dataset/nn_dataset/preprocessed/labelsTr'
for i in range(0,len(imges_path)):
    img=imges_path[i]
    labels=labels_path[i]
    print(img)
    pat_name=img.split('\\')[-1][0:8]
    img_obj=nib.load(img)
    label_obj=nib.load(labels)
    nib.save(img_obj,os.path.join(pathsaveim,pat_name+'_0000.nii.gz'))
    nib.save(label_obj,os.path.join(pathsavegt,pat_name+'.nii.gz'))
    #break

#%% nnunetv2 dataset conversion for cmrx dataset
import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from glob import glob
data_dir = 'C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/MICCAI_MBAS_2024/MBAS_Dataset/nn_dataset/preprocessed'
case_ids = glob(os.path.join(data_dir,'imagesTr','*.nii.gz'))
case_idslabels = glob(os.path.join(data_dir,'labelsTr','*.nii.gz'))

task_id = 220
task_name = "MBAS"

foldername = "Dataset%03.0d_%s" % (task_id, task_name)
nnUNet_raw='C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/MICCAI_MBAS_2024/MBAS_Dataset/nn_dataset/preprocessed/nnUNet_raw'
# setting up nnU-Net folders
out_base = join(nnUNet_raw, foldername)
imagestr = join(out_base, "imagesTr")
labelstr = join(out_base, "labelsTr")
maybe_mkdir_p(imagestr)
maybe_mkdir_p(labelstr)
################### iterate images and shitl in imagestr folder####################
for i in range(0,len(case_ids)):
    pathimgs=case_ids[i]
    pathmasks=case_idslabels[i]
    shutil.copy(pathimgs,imagestr)
    shutil.copy(pathmasks,labelstr)
    #break
  
generate_dataset_json(
    str(out_base),
    channel_names={
        0: "MRI",
    },
    labels={
        "background": 0,
        "RAC": 1, ### Right Atrium Cavity
        "LAC": 2, ###### Left Atrium Cavity
        "LRW": 3, ############## Left and right atrium walls
    },
    file_ending=".nii.gz",
    num_training_cases=len(case_ids),
)