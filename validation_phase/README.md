# validation instruction to run the code


## Installation instructions
git clone https://github.com/MIC-DKFZ/nnUNet.git

cd nnUNet

pip install -e .

### for testing simply 

pip install nnunetv2

## Dataset conversion
### please change input and output path of your system
python3 preprocess_nnunet_conversion_MBAS2024.py
## Dataset preprocessing

nnUNetv2_plan_and_preprocess -d 220 --verify_dataset_integrity

## Training

nnUNetv2_train 232 3d_fullres all

## testing

