# validation instruction to run the code


## Installation instructions
git clone https://github.com/MIC-DKFZ/nnUNet.git

cd nnUNet

pip install -e .

### for testing simply 

pip install nnunetv2

## Dataset conversion
### please change input and output path of your system
> python3 preprocess_nnunet_conversion_MBAS2024.py
## Dataset preprocessing

nnUNetv2_plan_and_preprocess -d 220 --verify_dataset_integrity

## Training

nnUNetv2_train 232 3d_fullres all

## testing

############### output where you put the prediction output########################
input dataset directory

inside test_folder we have subject folder like  MBAS_071 and inside subject folde file name should be MBAS_071_gt.nii.gz
             

resources_dir = '/home/aqayyum/docker_mine/MBAS_docker/val_docker/resources'  # Replace with your actual path

> python prediction.py --gpu 0 --input_dir /home/aqayyum/docker_mine/MBAS_docker/val_docker/test_folder/ --output_dir /home/aqayyum/docker_mine/MBAS_docker/val_docker/output/

## docker insstructions

##### Instructions to run docker for MBAS2024 challenges

################ build the docker##################################

docker pull abdulenib/cemrg_mbas_test:latest

###################### sometime we need permission to write the directory

chmod -R 777 /home/aqayyum/docker_mine/MBAS_docker/val_docker/output/ ################### this the output of your local machine #####################

################################################# run the docker #####################

docker pull abdulenib/cemrg_mbas_val:latest

docker run --rm  --gpus '"device=0"' -v /home/aqayyum/docker_mine/MBAS_docker/val_docker/test_folder/:/input -v /home/aqayyum/docker_mine/MBAS_docker/val_docker/output/:/output -it abdulenib/cemrg_mbas_val:latest

