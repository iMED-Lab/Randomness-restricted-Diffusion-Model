import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    if size_average:
        noise = torch.Tensor(np.random.normal(0, 0.01, img1.size())).cuda(img1.get_device())
        img1 = clip_by_tensor(img1 + noise, torch.Tensor(np.zeros(img1.size())).cuda(img1.get_device()),
                              torch.Tensor(np.ones(img1.size())).cuda(img1.get_device()))
        img2 = clip_by_tensor(img2 + noise, torch.Tensor(np.zeros(img2.size())).cuda(img2.get_device()),
                              torch.Tensor(np.ones(img2.size())).cuda(img2.get_device()))
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    a = 1
    b = 1
    c = 1

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    # ssim_1 = (2*mu1_mu2+C1)/(mu1_sq+mu2_sq+C1)
    # ssim_2 = (2*((sigma1_sq*sigma2_sq).pow(0.5))+C2)/(sigma1_sq+sigma2_sq+C2)
    ssim_3 = (sigma12+C2/2)/(((sigma1_sq*sigma2_sq).pow(0.5))+C2/2)

    if size_average:
        return ssim_3.mean()
    else:
        return ssim_map.mean()

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
