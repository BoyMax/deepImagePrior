import numpy as np
import torch
import torch.nn as nn 

class Convolution(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''
    def __init__(self, n_planes, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Convolution, self).__init__()
        
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1/2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1./np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in [ 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'
            
            
        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        
        convolution = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=1, padding=0)
        convolution.weight.data[:] = 0
        convolution.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            convolution.weight.data[i, i] = kernel_torch       
        self.convolution_ = convolution

        if preserve_size:

            if  self.kernel.shape[0] % 2 == 1: 
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - 1) / 2.)
                
            self.padding = nn.ReplicationPad2d(pad)
        
        self.preserve_size = preserve_size
    
    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x= input
        self.x = x
        return self.convolution_(x)
        
def get_kernel(kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    
    if phase == 0.5 and kernel_type != 'box': 
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    
        
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1./(kernel_width * kernel_width)
        
    elif kernel_type == 'gauss': 
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        
        center = (kernel_width + 1.)/2.
        print(center, kernel_width)
        sigma_sq =  sigma * sigma
        
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center)/2.
                dj = (j - center)/2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)        
    else:
        assert False, 'wrong method name'
    
    kernel /= kernel.sum()
    
    return kernel

#a = Downsampler(n_planes=3, factor=2, kernel_type='lanczos2', phase='1', preserve_size=True)






#################
# Learnable downsampler

# KS = 32
# dow = nn.Sequential(nn.ReplicationPad2d(int((KS - factor) / 2.)), nn.Conv2d(1,1,KS,factor))
    
# class Apply(nn.Module):
#     def __init__(self, what, dim, *args):
#         super(Apply, self).__init__()
#         self.dim = dim
    
#         self.what = what

#     def forward(self, input):
#         inputs = []
#         for i in range(input.size(self.dim)):
#             inputs.append(self.what(input.narrow(self.dim, i, 1)))

#         return torch.cat(inputs, dim=self.dim)

#     def __len__(self):
#         return len(self._modules)
    
# downs = Apply(dow, 1)
# downs.type(dtype)(net_input.type(dtype)).size()
