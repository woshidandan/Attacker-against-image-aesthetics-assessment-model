import numpy as np
from .utils import lrelu, rgb2lum, tanh_range, lerp
import torch
import math

device = torch.device("cuda:0" )

batch_size=32
nc=3


def EF(img,param):
    param=tanh_range(-3.5, 3.5, initial=0)(param.cpu().detach())
    param=param.to(device)
    return img * torch.exp(param[:, None, None, None] * np.log(2))


def color_temper_batch(img, paramr, paramg, paramb):

    bAve = torch.mean(torch.mean(img[:, :, :, 0], dim=1), dim=1) * paramr * 0.1
    gAve = torch.mean(torch.mean(img[:, :, :, 1], dim=1), dim=1) * paramr * 0.1  #应该是R
    rAve = torch.mean(torch.mean(img[:, :, :, 2], dim=1), dim=1) * paramr * 0.1
  # 2计算每个通道的增益系数
    bCoef = (bAve + gAve + rAve) / bAve
    gCoef = (bAve + gAve + rAve) / gAve
    rCoef = (bAve + gAve + rAve) / rAve

    Coef = torch.stack((bCoef.reshape(-1, 1, 1), gCoef.reshape(-1, 1, 1), rCoef.reshape(-1, 1, 1)), 0) / 3
    # print("temp shape :",temp1.reshape(half_batch_size, 1,1,nc).shape)
    Coef = Coef.reshape(1, 1, 1, nc)
    #print("img shape:",img.shape)
    #print("coef shape:",Coef.shape)
    out_final = torch.mul(img, Coef)

    return out_final

def updateAlpha(img, x):

    #x = x.reshape(-1, 1, 1, 1)
    # print("x a shape:", x_reshaped.shape)
    #print("x :",x)
    img=img *x
    #print(" img shape:",img.shape)
    out = torch.clamp( img, -1.0, 1.0)
    return out


def updateBeta(img, x):

    #x = x.reshape(-1, 1, 1, 1)
    #print("x", x[8])
    out = torch.clamp(x + img, -1.0, 1.0)

    return out

def GF(img,param):
    #log_gamma_range = np.log(3)
    param=torch.exp(tanh_range(-np.log(3), np.log(3))(param.cpu().detach()))
    ones=torch.ones(img.shape)*0.001
    ones=ones.to(device)
    param=param.to(device)
    return torch.pow(torch.maximum(img, ones), param[:, None, None, None])


def IWB(img,param):
    log_wb_range = 0.5
    mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
    mask_tensor = torch.from_numpy(mask).to(device)
    # print(features.shape)
    # assert mask.shape == (1, 3)
    param = param * mask_tensor
    param = torch.exp(tanh_range(-log_wb_range, log_wb_range)(param.cpu().detach()))
    # print("color sacling shape 1  :", color_scaling.shape)
    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    param *= 1.0 / (1e-5 + 0.27 * param[:, 0] + 0.67 * param[:, 1] +0.06 * param[:, 2])[:, None]
    param=param.to(device)
    return img * param[:, None, None, :]


def VF(img,param):
    param=torch.sigmoid(param.cpu().detach())
    param=param.to(device)
    return img + param[:, None, None, None] * 0.01


def CF(img,param):
    param=torch.tanh(param.cpu().detach())
    #ones=torch.ones(64,128,128,3)
    #zeros=torch.zeros_like(64,128,128,3)
    #luminance = torch.minimum(torch.maximum(rgb2lum(img), zeros), ones)
    luminance = 0.27 * img[:, :, :, 0] + 0.67 * img[:, :, :,1] + 0.06 * img[:, :, :, 2]
    luminance=torch.clamp(luminance, 0, 1)
    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    #print("contrast shape:",contrast_lum.shape)
    #print("lim shape:",luminance.shape)
    #print("img shape:",img.shape)
    contrast_image = img / (luminance[:,:,:,None] + 1e-6) * contrast_lum[:,:,:,None]
    param=param.to(device)
    #print("contrast shape:",contrast_image.shape)
    #print("param shape:",param.shape)
    return lerp(img, contrast_image, param[:, None, None, None])



def WNB(img,param):
    param=torch.sigmoid(param.cpu().detach())
    luminance = rgb2lum(img)
    param=param.to(device)
    # print("lim shape:",luminance.shape)
    # print("param shape:",param.shape)
    return lerp(img, luminance, param[:, None, None, None])


def LF(img,param):
    param=torch.sigmoid(param.cpu().detach())
    lower = param[:, 0]
    lower=lower.to(device)
    upper = param[:, 1] + 1
    upper=upper.to(device)
    #lower = lower[:, None, None, None]
    #upper = upper[:, None, None, None]
    return torch.clamp((img - lower[:, None, None, None]) / (upper[:, None, None, None] - lower[:, None, None, None] + 1e-6), 0.0, 1.0)


"""""
def SPF(img,param):
    param=torch.sigmoid(param.cpu().detach())
    ones=torch.ones(img.shape)
    img = torch.minimum(img, ones)
    print("img shape:",img.shape)
    hsv = R2H.rgb_to_hsv(img)
    print("hsv shape:",hsv.shape)
    s = hsv[:, :, :, 1:2]
    v = hsv[:, :, :, 2:3]
    # enhanced_s = s + (1 - s) * 0.7 * (0.5 - tf.abs(0.5 - v)) ** 2
    enhanced_s = s + (1 - s) * (0.5 - torch.abs(0.5 - v)) * 0.8
    print("echanced shape:",enhanced_s.shape)
    hsv1 = torch.cat([hsv[:, :, :, 0:1], enhanced_s, hsv[:, :, :, 2:]], axis=3)
    print("hsv shape:",hsv1.shape)
    full_color = R2H.hsv_to_rgb(hsv1)
    #param = param[:, None, None, None]
    #color_param = param
    #img_param = 1.0 - param
    print("param shape",param.shape)
    print("full color shape:",full_color.shape)
    return img * (1.0 - param[:, None, None, None]) + full_color * param[:, None, None, None]


"""


def dicrease_color(img,param):
    out = img*256
    realm=256*param
    out = out // realm * realm + realm*0.5
    return out/256