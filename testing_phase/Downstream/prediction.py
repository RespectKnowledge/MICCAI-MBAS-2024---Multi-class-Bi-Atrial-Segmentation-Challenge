# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:50:55 2024

@author: aq22
"""

import os
import torch
import argparse
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import time
#from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from os.path import join
import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--input_dir', type=str, default='/input', help='Path to input directory.')
    parser.add_argument('--output_dir', type=str, default='/output', help='Path to output directory.')
    #parser.add_argument('--model_pth', type=str, default='./save_pths/ABC_test.pth', help='Model saved path.')  #
    args = parser.parse_args()  # Corrected method call
    start_time = time.time()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('Generated new output directory:', args.output_dir)
    
    # Hard-coded checkpoint directory path
    resources_dir = '/home/aqayyum/docker_mine/MBAS_docker/resources'  # Replace with your actual path

    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Initialize the network architecture and load the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(resources_dir, 'nnUNet_results/Dataset220_MBAS/nnUNetTrainerUxLSTMBot__nnUNetPlans__3d_fullres'),
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )

    # Iterate over each file in the input directory
    for subdir, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('_gt.nii.gz'):
                file_path = os.path.join(subdir, file)
                
                # Prepare the input files as a list of lists
                #all_cases = [[file_path]]  # List of lists containing input file paths for each case

                # Define the output file path
                #pred_file_name = file.replace('_gt', '_label')
                #output_file = os.path.join(args.output_dir, pred_file_name)
                #output_files = [output_file]  # List of output file paths, one for each case

                # # Call the predictor with the list of cases and output files
                # predictor.predict_from_files(
                #     all_cases,                   # List of lists containing input file paths for each case
                #     output_files,                # List of output file paths, one for each case
                #     save_probabilities=False,
                #     overwrite=False,
                #     num_processes_preprocessing=1,
                #     num_processes_segmentation_export=1,
                #     folder_with_segs_from_prev_stage=None,
                #     num_parts=1,
                #     part_id=0
                # )
                img, props = SimpleITKIO().read_images([file_path])
                print("img.shape: ", img.shape)
                pred_array = predictor.predict_single_npy_array(img, props, None, None, False)
                pred_array = pred_array.astype(np.uint8)
                print("pred_array.shape: ", pred_array.shape)
                print("pred_array_labels_before: ", np.unique(pred_array))
                image = sitk.GetImageFromArray(pred_array)
                image.SetDirection(props['sitk_stuff']['direction'])
                image.SetOrigin(props['sitk_stuff']['origin'])
                image.SetSpacing(props['sitk_stuff']['spacing'])
                #image = sitk.Cast(image, sitk.sitkUInt8)
                pred_file_name = file.replace('_gt', '_label')
                output_file_name = os.path.join(args.output_dir, pred_file_name)
                sitk.WriteImage(
                    image,
                    output_file_name,
                    useCompression=True,)
                                 
                #print('Saved!!!')
                #print(f"Prediction saved for {file} at {output_file}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for prediction: {elapsed_time:.4f} seconds")

    print('Generate finished!')

if __name__ == '__main__':
    main()
