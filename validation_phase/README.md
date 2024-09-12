# Validation instruction to run the code for MBAS2024 challenge

## Installation instructions
git clone https://github.com/MIC-DKFZ/nnUNet.git

cd nnUNet

pip install -e .

### For testing simply 

pip install nnunetv2

## Dataset conversion
### Please change input and output path of your system
> python3 preprocess_nnunet_conversion_MBAS2024.py

## Dataset preprocessing

> nnUNetv2_plan_and_preprocess -d 220 --verify_dataset_integrity

## Training

> nnUNetv2_train 220 3d_fullres all

## Testing

> pip install -r requirements.txt

############ Please download trained weight and put inside the following folder path
resources/nnUNet_results/Dataset220_MBAS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all

https://mega.nz/file/ULVmFBxL#ErooV4REjK9ezOkCbf2jeXwGIw3Lf2RgjWrVboEvU_k

############### output where you put the prediction output########################
Input dataset directory

Inside test_folder we have subject folder like  MBAS_071 and inside subject folde file name should be MBAS_071_gt.nii.gz
             

resources_dir = '/home/aqayyum/docker_mine/MBAS_docker/val_docker/resources'  # Replace with your actual path

> python prediction.py --gpu 0 --input_dir /home/aqayyum/docker_mine/MBAS_docker/val_docker/test_folder/ --output_dir /home/aqayyum/docker_mine/MBAS_docker/val_docker/output/

## Docker instructions

##### Instructions to run docker for MBAS2024 challenge #############

################ build the docker##################################

docker pull abdulenib/cemrg_mbas_test:latest

###################### sometime we need permission to write the output directory ########
######## this the output of your local machine ######################################

> chmod -R 777 /home/aqayyum/docker_mine/MBAS_docker/val_docker/output/  

########################## pull docker image #####################

> docker pull abdulenib/cemrg_mbas_val:latest

######################## run docker #####################

> docker run --rm  --gpus '"device=0"' -v /home/aqayyum/docker_mine/MBAS_docker/val_docker/test_folder/:/input -v /home/aqayyum/docker_mine/MBAS_docker/val_docker/output/:/output -it abdulenib/cemrg_mbas_val:latest


## Acknowledgement
https://github.com/MIC-DKFZ/nnUNet
