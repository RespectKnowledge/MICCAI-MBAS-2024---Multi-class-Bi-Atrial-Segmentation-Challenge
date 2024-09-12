# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 10:57:50 2024

@author: aq22
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
import SimpleITK as sitk
import json
import os
import argparse
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--a_min", default=0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=125, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=20, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=256, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=256, type=int, help="roi size in z direction")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
parser.add_argument("--datasets")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--invis_patches", action="store_true", help="calculate loss on masked patches")
args = parser.parse_args()


import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from monai import transforms
import SimpleITK as sitk
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.UxLSTMBot_3d import get_uxlstm_bot_3d_from_plans
from nnunetv2.nets.UxLSTMBot_2d import get_uxlstm_bot_2d_from_plans

# Define ContrastiveLoss class
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, teacher_features, student_features):
        # Normalize features
        teacher_features = nn.functional.normalize(teacher_features, dim=1)
        student_features = nn.functional.normalize(student_features, dim=1)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(student_features, teacher_features.T) / self.temperature
        # Compute loss
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        return loss

# Define nnUNetTrainerUxLSTMBot class
class nnUNetTrainerUxLSTMBot(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        if len(configuration_manager.patch_size) == 2:
            model = get_uxlstm_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_uxlstm_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UxLSTMBot: {}".format(model))
        return model

# Define FeatureExtractor class
class FeatureExtractor:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = {name: None for name in layer_names}
        
        # Register hooks for the specified layers
        self.hooks = [self._register_hook(name) for name in layer_names]

    def _register_hook(self, layer_name):
        def hook_fn(module, input, output):
            self.features[layer_name] = output
        
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module.register_forward_hook(hook_fn)
        raise ValueError(f"Layer '{layer_name}' not found in the model.")

    def get_features(self):
        return self.features

# Generate encoder layers
def generate_encoder_layers(num_stages, num_blocks):
    layers = []
    for stage in range(num_stages):
        for block in range(num_blocks):
            layers += [
                f'encoder.stages.{stage}.{block}.conv1',
                f'encoder.stages.{stage}.{block}.conv2',
                f'encoder.stages.{stage}.{block}.conv3'
            ]
    return layers

# Define target layers
num_stages = 7
num_blocks = 1
encoder_layers = generate_encoder_layers(num_stages, num_blocks)

xlstm_layers = [
    #'xlstm.norm',
    'xlstm.vil.layer.proj_up',
    'xlstm.vil.layer.q_proj',
    'xlstm.vil.layer.k_proj',
    'xlstm.vil.layer.v_proj',
    'xlstm.vil.layer.conv1d.conv',
    #'xlstm.vil.layer.mlstm_cell'
]

target_layers = encoder_layers + xlstm_layers
#target_layers = xlstm_layers
#target_layers = encoder_layers
print("All Target Layers:")
print(target_layers)

# Load model and configuration
#model_training_output_dir = 'C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/Bond_HI/nnUNetTrainerUxLSTMEnc__nnUNetPlans__3d_fullres'
model_training_output_dir = '/home/aqayyum/SSL_xLSTM_model/dataset/'
dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
plans = load_json(join(model_training_output_dir, 'plans.json'))
plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration('3d_fullres')
#num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

# Load models
teacher_model = get_uxlstm_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager, 1, deep_supervision=False)
student_model = get_uxlstm_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager, 1, deep_supervision=False)

# Create FeatureExtractor instances for both Teacher and Student models
feature_extractor_teacher = FeatureExtractor(teacher_model, target_layers)
feature_extractor_student = FeatureExtractor(student_model, target_layers)

class BasicDataset(Dataset):
    def __init__(self, data_list, args, phase='train'):
        super(BasicDataset, self).__init__()
        self.data_list = data_list

        self.train_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.load_image(self.data_list[item])
        if self.phase == 'train':
            data = self.train_transform(data)
        elif self.phase == 'val':
            data = self.val_transform(data)
        return data
    
    def load_image(self, file_dic):
        image_path = file_dic['image']
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        image = np.moveaxis(image, (0, 1, 2), (2, 1, 0))
        return {'image': image,
                'image_path': image_path}

        
import os
from glob import glob
from monai import transforms

import monai.transforms as transforms
import numpy as np

class PretrainDatasetMRI(BasicDataset):
    def __init__(self, data_list, args, phase='train'):
        super(PretrainDatasetMRI, self).__init__(data_list, args, phase='train')
        
        self.data_list = data_list

        self.train_transform = transforms.Compose(
            [
                transforms.Lambdad(keys=["image"], func=lambda x: x.astype(np.float32)),  # Convert to float32
                transforms.EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                transforms.SpatialPadd(keys="image",
                                       spatial_size=[args.roi_x, args.roi_y, args.roi_z],
                                       mode='symmetric'),
                transforms.RandSpatialCropd(keys="image",
                                            roi_size=[args.roi_x, args.roi_y, args.roi_z]),
                transforms.RandSpatialCropSamplesd(
                    keys=['image'],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Lambdad(keys=["image"], func=lambda x: x.astype(np.float32)),  # Convert to float32
                transforms.EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.SpatialPadd(keys="image",
                                       spatial_size=[args.roi_x, args.roi_y, args.roi_z],
                                       mode='symmetric'),
                transforms.RandSpatialCropd(keys="image",
                                            roi_size=[args.roi_x, args.roi_y, args.roi_z]),
                transforms.RandSpatialCropSamplesd(
                    keys=['image'],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.phase = phase

# Set paths and load data
#pathdata = '/home/aqayyum/SSL_xLSTM/'
#pathdata = 'C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/MICCAI_MBAS_2024/MBAS_Dataset/nn_dataset/preprocessed'
pathdata='/home/aqayyum/SSL_xLSTM_model/main_data/'
pathimg = glob(os.path.join(pathdata, 'imagesTr', '*.nii.gz'))
train_files = [{'image': image_name} for image_name in pathimg]
# pathimg = glob(os.path.join(pathdata, 'imagesTr', '*.nii.gz'))
# train_files = [{'image': image_name} for image_name in pathimg]

# Define args
#args = Args()

# Initialize datasets
# train_datasets = PretrainDatasetMRI(train_files, args, phase='train')
# valid_datasets = PretrainDatasetMRI(train_files, args, phase='val')

# print(len(train_datasets))
# print(len(valid_datasets))

#data_list=
from glob import glob
#pathdata='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/'
#pathdata='/home/aqayyum/SSL_xLSTM/'
#pathimg=glob(os.path.join(pathdata,'imagesTr','*.nii.gz'))
#pathlabel=glob(os.path.join(pathdata,'labels','*.nii.gz'))
#train_files=[{'image':image_name} for image_name in pathimg]

train_datasets=PretrainDatasetMRI(train_files,args, phase='train')
valid_datasets=PretrainDatasetMRI(train_files,args, phase='val')
print(len(train_datasets))
print(len(valid_datasets))
train_dataloader = DataLoader(train_datasets,
                                  batch_size=2,
                                  num_workers=0,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=True)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to device
teacher_model.to(device)
student_model.to(device)
student_model.train()

# Define optimizer
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# Initialize contrastive loss
contrastive_loss = ContrastiveLoss().to(device)

num_epochs = 200

#%
# Initialize variables
best_loss = float('inf')
save_path = 'best_student_model_enco_bottom.pth'

# Training loop
for epoch in range(num_epochs):
    student_model.train()
    teacher_model.eval()  # Set teacher model to evaluation mode
    
    epoch_loss = 0  # To accumulate loss over the epoch
    
    for i, d in enumerate(train_dataloader):
        #print(d[0]["image"].dtype)  # Check data type before .float()
        data1 = d[0]["image"].to(device).float()
        data2 = d[1]["image"].to(device).float()
        print(f'Epoch {i}, Data1 dtype: {data1.dtype}, Data2 dtype: {data2.dtype}')
    
        # # If you notice anything unusual, stop the training loop.
        # if data1.dtype != torch.float32 or data2.dtype != torch.float32:
        #     print("Unexpected data type!")
        #     break
        
        optimizer.zero_grad()
        
        # Forward pass through teacher model
        with torch.no_grad():
            teacher_output = teacher_model(data1)
            teacher_features = feature_extractor_teacher.get_features()
            if any(features is None for features in teacher_features.values()):
                print("Some Teacher features are None!")
                continue
        
        # Forward pass through student model
        student_output = student_model(data2)
        student_features = feature_extractor_student.get_features()
        if any(features is None for features in student_features.values()):
            print("Some Student features are None!")
            continue
        
        # Ensure features are not None
        if any(features is None for features in teacher_features.values()) or \
           any(features is None for features in student_features.values()):
            print(f"Skipping batch due to NoneType features")
            continue
        
        # Initialize variable to accumulate total loss
        total_loss = 0
        
        # Compute contrastive loss for each layer
        for layer in target_layers:
            teacher_layer_features = teacher_features[layer].flatten(start_dim=1)
            student_layer_features = student_features[layer].flatten(start_dim=1)
            
            # Compute contrastive loss for this layer
            cl_loss = contrastive_loss(teacher_layer_features, student_layer_features)
            print(f'Contrastive Loss for layer {layer}: {cl_loss.item()}')
            
            # Accumulate total loss
            total_loss += cl_loss
        
        # Accumulate epoch loss
        epoch_loss += total_loss.item()
        
        # Backward pass and optimization
        total_loss.backward()  # No need for retain_graph=True here
        optimizer.step()
    
    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}] completed with avg loss: {avg_epoch_loss:.4f}')
    
    # Check if the current epoch's loss is the best
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(student_model.state_dict(), save_path)
        print(f'Best model saved with loss: {best_loss:.4f}')


