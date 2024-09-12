# Testing phase code for Multi-class-Bi-Atrial-Segmentation-Challenge

## Installation
Requirements: Ubuntu 20.04, CUDA 11.8

1. Create a virtual environment: conda create -n uxlstm python=3.10 -y and conda activate uxlstm 
2. Install Pytorch 2.0.1: pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
3. Download code: git clone https://github.com/tianrun-chen/xLSTM-UNet-PyTorch.git
4. cd xLSTM-UNet-PyTorch/UxLSTM and run pip install -e .


# Student teacher model SSL part
Go to folder SSL_student_teacher and run the following command

> python3 xlstm_bottom_ssl_updated1

## trained weight of Student Teacher model can be found here
https://mega.nz/file/8L90HTCS#1vhrsZ1NSHeXkNaygcd9Hx15t1oq8fFLMLUaJJslhF0

# Downstream part
## Dataset conversion

> python3 preprocess_nnunet_conversion_MBAS2024.py

## Dataset preprocessing

> nnUNetv2_plan_and_preprocess -d 220 --verify_dataset_integrity

## Training

Go to folder /home/aqayyum/xLSTM-UNet-PyTorch/UxLSTM/nnunetv2/run add this file into this folder run_finetuning_xLSTM_bottom_model.py
then run the following command.
this is my folder /home/aqayyum/xLSTM-UNet-PyTorch/data to add best_student_model_enco_bottom.pth trained student teacher model

> CUDA_VISIBLE_DEVICES=0 python3 /home/aqayyum/xLSTM-UNet-PyTorch/UxLSTM/nnunetv2/run/run_finetuning_xLSTM_bottom_model.py 220 3d_fullres all -pretrained_weights /home/aqayyum/xLSTM-UNet-PyTorch/data/best_student_model_enco_bottom.pth -tr nnUNetTrainerUxLSTMBot -lr 0.001 -bs 2

## Prediction
donwload the trained weight and put in fold_all folder
https://mega.nz/file/tGs3TKDK#HI_aOR0j_pH7kqIxzvvVXE55E8etNQRmSAE5Kl-Ikdg

> pip install -r requirements.txt

resources_dir = '/home/aqayyum/docker_mine/MBAS_docker/resources' change your path
> python prediction.py --gpu 0 --input_dir /home/aqayyum/docker_mine/MBAS_docker/test_folder/ --output_dir /home/aqayyum/docker_mine/MBAS_docker/output/

## Docker instructions

> docker pull abdulenib/cemrg_mbas_test:latest

###################### sometime we need permission to write the directory ##############################

> chmod -R 777 /home/aqayyum/docker_mine/MBAS_docker/val_docker/output/ 

################################################# run the docker #####################


> docker run --rm  --gpus '"device=0"' -v /home/aqayyum/docker_mine/MBAS_docker/test_folder/:/input -v /home/aqayyum/docker_mine/MBAS_docker/output/:/output -it abdulenib/cemrg_mbas_test:latest

## Acknowledgement

https://github.com/tianrun-chen/xLSTM-UNet-PyTorch

https://github.com/MIC-DKFZ/nnUNet


