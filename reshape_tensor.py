import torch
import numpy as np

def reshape_patch(img_tensor, patch_size):
    #[3,20,1,64,64,]
    batch_size = (img_tensor.shape)[0]
    seq_length = (img_tensor.shape)[1]
    img_height = (img_tensor.shape)[3]
    img_width = (img_tensor.shape)[4]
    num_channels = (img_tensor.shape)[2]
    a = torch.reshape(img_tensor, (batch_size, seq_length, num_channels,
                                int(img_height/patch_size), patch_size,
                                int(img_width/patch_size), patch_size,
                                )) #BSNHPWP
    b = a.permute(0,1,4,6,2,3,5) ##?? to be check #BSPPNHW
    patch_tensor = torch.reshape(b, (batch_size, seq_length,
                                  patch_size*patch_size*num_channels,
                                  int(img_height/patch_size),
                                  int(img_width/patch_size)))
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    batch_size = int((patch_tensor.shape)[0])
    seq_length = int((patch_tensor.shape)[1])
    patch_height = int((patch_tensor.shape)[3])
    patch_width = int((patch_tensor.shape)[4])
    channels = int((patch_tensor.shape)[2])
    img_channels = int(channels / (patch_size*patch_size))
    
    a = torch.reshape(patch_tensor, (batch_size, seq_length,
                                  patch_size, patch_size,
                                  img_channels,
                                  patch_height, patch_width)) #BSPPNHW
    b = a.permute(0,1,4,5,2,6,3) #BSNHPWP
    img_tensor = torch.reshape(b, (batch_size, seq_length,
                                img_channels,
                                patch_height * patch_size,
                                patch_width * patch_size
                                  )) #BSNHW
    return img_tensor


