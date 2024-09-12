# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 09:19:46 2024

@author: aq22
"""

from unittest.mock import patch
from run_training import run_training_entry
import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

#import torch
#from torch._dynamo import OptimizedModule
#from torch.nn.parallel import DistributedDataParallel as DDP

#def load_pretrained_weights_xsltm_SSL(network, fname, verbose=False, freeze_encoder_layers=False):
#    """
#    Transfers weights from a pretrained model to the network, optionally freezing encoder layers.
#    
#    Args:
#        network: The model to load weights into.
#        fname: Path to the file containing pretrained weights.
#        verbose: If True, prints details about the loaded weights.
#        freeze_encoder_layers: If True, freezes all encoder layers after loading weights.
#    """
#    saved_model = torch.load(fname)
#    #pretrained_dict = saved_model['network_weights']
#    pretrained_dict = saved_model
#
#    # No skip strings; load all encoder weights
#    skip_strings_in_pretrained = []
#
#    if isinstance(network, DDP):
#        mod = network.module
#    else:
#        mod = network
#    if isinstance(mod, OptimizedModule):
#        mod = mod._orig_mod
#
#    model_dict = mod.state_dict()
#
#    # Verify that all but the skip patterns have the same shape
#    for key, _ in model_dict.items():
#        if not any(skip_str in key for skip_str in skip_strings_in_pretrained):
#            assert key in pretrained_dict, \
#                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
#                f"compatible with your network."
#            assert model_dict[key].shape == pretrained_dict[key].shape, \
#                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
#                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
#                f"does not seem to be compatible with your network."
#
#    # Filter the pretrained dictionary to include all keys
#    filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items()
#                                if k in model_dict.keys()}
#
#    # Update the model's state_dict with the filtered pretrained weights
#    model_dict.update(filtered_pretrained_dict)
#
#    # Load the state_dict into the model
#    print("################### Loading pretrained weights from file ", fname, '###################')
#    if verbose:
#        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
#        for key, value in filtered_pretrained_dict.items():
#            print(key, 'shape', value.shape)
#        print("################### Done ###################")
#    mod.load_state_dict(model_dict)
#    
#    # Optionally freeze encoder layers
#    if freeze_encoder_layers:
#        for name, param in mod.named_parameters():
#            if any(prefix in name for prefix in ['encoder.stem.', 'encoder.stages.', 'encoder.xlstm_layers.']):
#                param.requires_grad = False  # Freezing the layers
#
#        print("################### Encoder layers have been frozen ###################")
#        if verbose:
#            for name, param in mod.named_parameters():
#                if not param.requires_grad:
#                    print(f"Layer {name} is frozen")


import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from unittest.mock import patch

def load_pretrained_weights_xsltm_SSLlatest(network, fname, verbose=False, freeze_encoder_layers=True):
    """
    Transfers weights from a pretrained model to the network, optionally freezing encoder layers.
    
    Args:
        network: The model to load weights into.
        fname: Path to the file containing pretrained weights.
        verbose: If True, prints details about the loaded weights.
        freeze_encoder_layers: If True, freezes all encoder layers after loading weights.
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # Filter the pretrained dictionary to include only keys that are in the model's state_dict
    filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Identify missing and mismatched keys
    missing_keys = set(model_dict.keys()) - set(filtered_pretrained_dict.keys())
    mismatched_keys = {k: (v.size(), model_dict[k].size()) for k, v in pretrained_dict.items() if k in model_dict and v.size() != model_dict[k].size()}

    if missing_keys:
        print("Missing keys in pretrained weights:", missing_keys)
    if mismatched_keys:
        print("Mismatched keys in pretrained weights:", mismatched_keys)

    # Update model_dict with filtered pretrained weights
    model_dict.update(filtered_pretrained_dict)

    # Load the state_dict into the model
    print("Loading pretrained weights from file", fname)
    mod.load_state_dict(model_dict)

    # Optionally freeze encoder layers
    if freeze_encoder_layers:
        for name, param in mod.named_parameters():
            if any(prefix in name for prefix in ['xlstm.vil.layer.proj_up', 'xlstm.vil.layer.q_proj', 'xlstm.vil.layer.k_proj','xlstm.vil.layer.v_proj','xlstm.vil.layer.conv1d.conv']):
                param.requires_grad = True

        print("Encoder layers have been has been finetuning")
        if verbose:
            for name, param in mod.named_parameters():
                if not param.requires_grad:
                    print(f"Layer {name} is frozen")
    
    # Return the model for further use
    return mod


if __name__ == '__main__':
    # Patch the run_training.load_pretrained_weights with the custom function
    with patch("run_training.load_pretrained_weights", load_pretrained_weights_xsltm_SSLlatest):
        run_training_entry()  # Start the training process

