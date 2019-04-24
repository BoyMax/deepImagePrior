
# coding: utf-8

# In[4]:


import numpy as np
import torch
import torch.nn as nn 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
from utils.sr_utils import *
from models import *


class BiConvolution(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''
    def __init__(self, n_planes, kernel_type, kernel_path,kernel,preserve_size=False):
        super(BiConvolution, self).__init__()
        
        if kernel_type ==  'udf':  # user_defined
            kernel_type_ = 'udf'

        else:
            assert False, 'wrong name kernel'
            
        self.kernel = kernel
        
        convolution = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=1,padding=0)
        convolution.weight.data[:] = 0
        convolution.bias.data[:] = 0
        kernel_torch = self.kernel
        for i in range(n_planes):
            convolution.weight.data[i, i] = kernel_torch       
        self.convolution_ = convolution

        if preserve_size:

            if  self.kernel.shape[0] % 2 == 1: 
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - 1) / 2.)
                
            self.padding = nn.ReflectionPad2d(pad)
        
        self.preserve_size = preserve_size
    
    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x= input
        self.x = x
        return self.convolution_(x)
    
def load_kernel(kernel_type,kernel_path):
    img_k_pil, img_k_np = get_image(kernel_path, -1)
    return img_k_np

