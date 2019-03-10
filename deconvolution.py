from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from models.convolution import Convolution

from utils.sr_utils import *

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark =False
#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor

# Load LR_image
path_LR_image = 'data/sr/Cameraman256_gauss.png'
path_HR_image = 'data/sr/Cameraman256.png'
img_lr_pil, img_lr_np = get_image(path_LR_image, -1)
img_hr_pil, img_hr_np = get_image(path_HR_image, -1)

#Set parameters
input_depth = 32
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='udf'
kernel_path ='../data/sr/kernel_gauss.png'
LR = 0.01
tv_weight = 0.0
OPTIMIZER = 'adam'
PLOT = True

# Get input(noise z)
net_input = get_noise(input_depth, INPUT, (img_hr_pil.size[1], img_hr_pil.size[0])).type(dtype).detach()

# Get net for noise(z)
NET_TYPE = 'skip' # UNet, ResNet
net = get_net(input_depth, 'skip', pad, n_channels=1, skip_n33d=128, skip_n33u=128, skip_n11=4, 
              num_scales=5, upsample_mode='bilinear').type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)
img_LR_var = np_to_torch(img_lr_np).type(dtype)

#convolution initialization for LR
convolution = Convolution(n_planes=1, kernel_type=KERNEL_TYPE, kernel_path, preserve_size=True).type(dtype)


# define closure and optimize
def closure():
    global i, net_input
    
    reg_noise_std = 0.01
    net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_HR = net(net_input)
    out_LR = convolution(out_HR)

    total_loss = mse(out_LR, img_LR_var) 
    
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)
        
    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(img_lr_np, torch_to_np(out_LR))
    psnr_HR = compare_psnr(img_hr_np, torch_to_np(out_HR))
    print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')
                      
    # History
    psnr_history.append([psnr_LR, psnr_HR])
    
    if PLOT and i % 100 == 0:
        out_HR_np = torch_to_np(out_HR)
        plot_image_grid([img_hr_np, np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)

    i += 1
    
    return total_loss

psnr_history = [] 
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
num_iter=2000
reg_noise_std = 0.03
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)