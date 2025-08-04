from .utils_for_filters import  rgb2lum, tanh_range, lerp
from .function_recycle import *
import kornia as K

device = torch.device("cuda:0" )


def get_max_min(tensor,name):
    fake_pred_flap = tensor.cpu().reshape(-1, 1)
    fake_max = fake_pred_flap[torch.argmax(fake_pred_flap, dim=0, keepdim=False)]
    fake_min = fake_pred_flap[torch.argmin(fake_pred_flap, dim=0, keepdim=False)]
    print("%s max %.6f:"%(name,fake_max))
    print("%s min %.6f:"%(name,fake_min))


def EF(param,img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    if find_zero(param, param.shape[0]):
        param = deal_with_zero(param,param.shape[0]).to(device)
    else:
        param = param.to(device)
    param=tanh_range(-1, 1, initial=0)(param.cpu().detach())
    param=param/2+0.5
    weight=torch.exp(param -0.7).to(device)
    result=img*weight[:,None,None,None]
    result=torch.clamp(result,min=0,max=1)
    if torch.isnan(result).any():
        print("EF nan")

    return result



def GF(param,img):
    if torch.isnan(img).any():
        print("GF input  nan")
    if find_zero(param, param.shape[0]):
        param = deal_with_zero(param,param.shape[0]).to(device)
    else:
        param = param.to(device)
    param=torch.exp(tanh_range(-np.log(3), np.log(3))(param.cpu().detach())).to(device)
    ones=torch.ones(img.shape)*0.001
    ones=ones.to(device)
    result=torch.pow(torch.maximum(img, ones), param[:, None, None, None])
    result = torch.clamp(result, min=0, max=1)
    return result



def IWB(param,img):
    if torch.isnan(img).any():
        print("IWB input  nan")
    if find_zero(param, param.shape[0]):
        param = deal_with_zero(param,param.shape[0]).to(device)
    else:
        param = param.to(device)
    log_wb_range = 0.5
    mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
    mask_tensor = torch.from_numpy(mask).to(device)
    param = param * mask_tensor
    param = torch.exp(tanh_range(-log_wb_range, log_wb_range)(param.cpu().detach())).to(device)
    param *= 1.0 / (1e-5 + 0.27 * param[:, 0] + 0.67 * param[:, 1] +0.06 * param[:, 2])[:, None]
    result=img * param[:, :, None, None]
    result = torch.clamp(result, min=0, max=1)
    return result


def VF(param,img):
    if torch.isnan(img).any():
        print("VF input  nan")
    if find_zero(param, param.shape[0]):
        param = deal_with_zero(param,param.shape[0]).to(device)
    else:
        param = param.to(device)
    param=torch.sigmoid(param)
    result=img + param[:, None, None, None] * 0.01
    result = torch.clamp(result, min=0, max=1)
    return result


def CF(param,img):
    if torch.isnan(img).any():
        print("CF input  nan")
    if find_zero(param, param.shape[0]):
        param = deal_with_zero(param,param.shape[0]).to(device)
    else:
        param = param.to(device)
    param=torch.tanh(param)
    luminance = 0.27 * img[:, :, :, 0] + 0.67 * img[:, :, :,1] + 0.06 * img[:, :, :, 2]
    luminance=torch.clamp(luminance, 0, 1)
    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance[:,:,:,None] + 1e-6) * contrast_lum[:,:,:,None].to(device)
    result=lerp(img, contrast_image, param[:, None, None, None])
    result = torch.clamp(result, min=0, max=1)
    return result



def WNB(param,img):
    if torch.isnan(img).any():
        print("WNB input  nan")
    if find_zero(param, param.shape[0]):
        param = deal_with_zero(param,param.shape[0]).to(device)
    else:
        param = param.to(device)
    param=torch.sigmoid(param)
    luminance = rgb2lum(img)
    result=lerp(img, luminance, param[:, None, None, None])
    result = torch.clamp(result, min=0, max=1)
    return result


def LF(param,img):
    if torch.isnan(img).any():
        print("LF input  nan")
    if find_zero(param, param.shape[0]):
        param = deal_with_zero(param,param.shape[0]).to(device)
    else:
        param = param.to(device)
    param=torch.sigmoid(param)
    lower = param[:, 0]
    upper = param[:, 0] + 1
    lower=lower.to(device)
    upper=upper.to(device)
    result=torch.clamp((img - lower[:, None, None, None]) / (upper[:, None, None, None] - lower[:, None, None, None] + 1e-6), 0.0, 1.0)
    result = torch.clamp(result, min=0, max=1)
    return result


def find_zero(tensor, size):
    for i in range(size):
        if tensor[i].item() == 0:
            # print("have 0")
            return True
    return False

def deal_with_zero(tensor, size):
    temp = torch.zeros([size, 1]).to(device)
    for i in range(size):
        if tensor[i].item() != 0:
            temp[i] = temp[i] + tensor[i]
        else:
            temp[i] = 1
    return temp

def get_fixed_filter(all_choice,filter_used):
    one_pace_id=torch.zeros(all_choice.shape[0])
    for i in range(all_choice.shape[0]):
        for j in range(16):
            values, indices=torch.topk(all_choice[i],(j+1))
            if filter_used[i][indices[-1]]==0:
                filter_used[i][indices[-1]] = filter_used[i][indices[-1]]+1
                one_pace_id[i]=one_pace_id[i]+indices[-1]
                break
    #print("one pace id:",one_pace_id)
    #print("filter used:",filter_used)
    return  one_pace_id



def adjust_contrast(adjust_contrast_param,img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    if find_zero(adjust_contrast_param, adjust_contrast_param.shape[0]):
        adjust_contrast_param = deal_with_zero(adjust_contrast_param,
                                               adjust_contrast_param.shape[0]).to(device)
    else:
        adjust_contrast_param = adjust_contrast_param.to(device)
    adjust_contrast_param = adjust_contrast_param * 0.999 * 2
    out = K.enhance.adjust_contrast(image=img, factor=adjust_contrast_param)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")
    return out

def adjust_brightness(adjust_brightness_param, img):
    if torch.isnan(img).any():
        print("adjust_brightness_fc input  nan")
    adjust_brightness_param = (adjust_brightness_param - 0.5) * 2
    if find_zero(adjust_brightness_param, adjust_brightness_param.shape[0]):
        adjust_brightness_param = deal_with_zero(adjust_brightness_param,
                                                 adjust_brightness_param.shape[0]).to(device)
    else:
        adjust_brightness_param = adjust_brightness_param.to(device)
    adjust_brightness_param = adjust_brightness_param  # 原函数是img+adjust_brightness_param,，这里应该将param缩小一点避免出现过亮的图片
    out = K.enhance.adjust_brightness(img, adjust_brightness_param)
    if torch.isnan(out).any():
        print("adjust_brightness_fc nan")

    return out

def adjust_saturation(adjust_saturation_param, img):
    if torch.isnan(img).any():
        print("adjust_saturation_fc input  nan")
    if find_zero(adjust_saturation_param, adjust_saturation_param.shape[0]):
        adjust_saturation_param = deal_with_zero(adjust_saturation_param,
                                                 adjust_saturation_param.shape[0]).to(device)
    else:
        adjust_saturation_param = adjust_saturation_param.to(device)
    out = K.enhance.adjust_saturation(img, adjust_saturation_param * 2)  # param范围（0，2），1表示不做修改
    if torch.isnan(out).any():
        print("adjust_saturation_fc nan")
    return out

def adjust_gamma(adjust_gamma_param, img):
    if torch.isnan(img).any():
        print("adjust_gamma_fc input  nan")
    if torch.isnan(img).any():
        print("adjust_gamma_fc input  nan")
    if find_zero(adjust_gamma_param, adjust_gamma_param.shape[0]):
        adjust_gamma_param = deal_with_zero(adjust_gamma_param, adjust_gamma_param.shape[0]).to(device)
    else:
        adjust_gamma_param = adjust_gamma_param.to(device)
    # print("adjust_gamma_param",adjust_gamma_param)
    out = K.enhance.adjust_gamma(img, gamma=adjust_gamma_param * 2)
    if torch.isnan(out).any():
        print("adjust_gamma_fc nan")
    return out

def adjust_hue(adjust_hue_param, img):
    if torch.isnan(img).any():
        print("adjust_hue_fc input  nan")
    if torch.isnan(img).any():
        print("adjust_hue_fc input  nan")
    adjust_hue_param = (adjust_hue_param - 0.5) * 2
    if find_zero(adjust_hue_param, adjust_hue_param.shape[0]):
        adjust_hue_param = deal_with_zero(adjust_hue_param, adjust_hue_param.shape[0]).to(device)
    else:
        adjust_hue_param = adjust_hue_param.to(device)
    out = K.enhance.adjust_hue(img, adjust_hue_param)
    if torch.isnan(out).any():
        print("adjust_hue_fc nan")
    return out

def blur_pool2d(blur_pool2d_param, img):
    if torch.isnan(img).any():
        print("blur_pool2d_fc input  nan")
    if find_zero(blur_pool2d_param, blur_pool2d_param.shape[0]):
        blur_pool2d_param = deal_with_zero(blur_pool2d_param, blur_pool2d_param.shape[0]).to(device)
    else:
        blur_pool2d_param = blur_pool2d_param.to(device)
    # print("param :", blur_pool2d_param )
    # print("param :",blur_pool2d_param%0.1)
    blur_pool2d_param = blur_pool2d_param * 0.5
    if blur_pool2d_param < 0.1:
        blur_pool2d_param = blur_pool2d_param + 0.1
    kernel_size = (blur_pool2d_param - (blur_pool2d_param % 0.1)).item()
    kernel_size = int(kernel_size * 10)
    # print("kernel:",kernel_size)
    #input_temp = img.reshape(1, 3, 256, 256)
    # print("input shape:", input_temp.shape)
    out = K.filters.blur_pool2d(img, kernel_size=kernel_size)

    # print("out_temp shape:", out_temp.shape)
    #out = transforms.functional.resize(out_temp, [256, 256])
    # out=out_temp.resize(3,256,256)
    # print("out shape:",out.shape)
    if torch.isnan(out).any():
        print("blur_pool2d_fc nan")
    return out

def sharpen(sharpen_param, img):
    print("img in sharp shape:",img.shape)
    print("img in sharpen_param shape:", sharpen_param.shape)
    if torch.isnan(img).any():
        print("sharpen_fc input  nan")
    if find_zero(sharpen_param, sharpen_param.shape[0]):
        sharpen_param = deal_with_zero(sharpen_param, sharpen_param.shape[0]).to(device)
    else:
        sharpen_param = sharpen_param.to(device)
    # print("sharpen shape:",sharpen_param.shape)
    sharpen = K.filters.UnsharpMask((9, 9), sigma=(sharpen_param[0] * 2, sharpen_param[0] * 2))
    #input_temp = img.detach().reshape([-1, 3, 256, 256])
    sharpened_tensor = sharpen(img)
    #sharpened_tensor = sharpened_tensor.detach().reshape(3, 256, 256)
    # print("sharpened_tensor shape",sharpened_tensor.shape)
    if torch.isnan(sharpened_tensor).any():
        print("sharpened_tensor nan")
    return  sharpened_tensor

def equalize_clahe(equalize_clahe_param, img):
    if torch.isnan(img).any():
        print("RandomPlasmaShadow_fc input  nan")
    if find_zero(equalize_clahe_param, equalize_clahe_param.shape[0]):
        equalize_clahe_param = deal_with_zero(equalize_clahe_param, equalize_clahe_param.shape[0]).to(
            device)
    else:
        equalize_clahe_param = equalize_clahe_param.to(device)
    equalize_clahe_param = equalize_clahe_param * 0.5
    if equalize_clahe_param < 0.1:
        equalize_clahe_param = equalize_clahe_param + 0.1
    grid_size = (equalize_clahe_param - (equalize_clahe_param % 0.1)).item()
    grid_size = int(grid_size * 10)
    temp = grid_size
    grid_size = [grid_size, grid_size]
    grid_size = tuple(grid_size)
    # print("grid size:",grid_size)
    out = K.enhance.equalize_clahe(img, grid_size=grid_size)
    # input_cpu=input.cpu()
    # print(input_cpu.device)
    if torch.isnan(out).any():
        print("RandomPlasmaShadow_fc nan")
    return out

def RandomPlanckianJitter(RandomPlanckianJitter_param, img):
    if torch.isnan(img).any():
        print("RandomPlanckianJitter_fc input  nan")
    if find_zero(RandomPlanckianJitter_param, RandomPlanckianJitter_param.shape[0]):
        RandomPlanckianJitter_param = deal_with_zero(RandomPlanckianJitter_param,
                                                     RandomPlanckianJitter_param.shape[0]).to(device)
    else:
        RandomPlanckianJitter_param = RandomPlanckianJitter_param.to(device)
    RandomPlanckianJitter_param = RandomPlanckianJitter_param * 0.1
    RPJ = K.augmentation.RandomPlanckianJitter(mode='CIED', p=RandomPlanckianJitter_param.cpu())
    out = RPJ(img.cpu()).to(device)
    if torch.isnan(out).any():
        print(" RandomPlanckianJitter_fc nan")
    return  out

def solarize(solarize_param, img):
    if torch.isnan(img).any():
        print("ColorJiggle_fc input  nan")
    if find_zero(solarize_param, solarize_param.shape[0]):
        solarize_param = deal_with_zero(solarize_param, solarize_param.shape[0]).to(device)
    else:
        solarize_param = solarize_param.to(device)
    # print("color shape:",ColorJiggle_param.shape)
    solarize_param = solarize_param
    out = K.enhance.solarize(img, thresholds=solarize_param)
    out = out.to(device)
    if torch.isnan(out).any():
        print("ColorJiggle_fc nan")
    # print("++++++++++++++++++++")
    # print("out shape:",out.shape)
    out = out.reshape(-1,3, 256, 256)
    return out

def RandomBoxBlur(RandomBoxBlur_param, img):
    if torch.isnan(img).any():
        print("RandomBoxBlur_fc input  nan")
    if find_zero(RandomBoxBlur_param, RandomBoxBlur_param.shape[0]):
        RandomBoxBlur_param = deal_with_zero(RandomBoxBlur_param, RandomBoxBlur_param.shape[0]).to(
            device)
    else:
        RandomBoxBlur_param = RandomBoxBlur_param.to(device)
    RandomBoxBlur_param = RandomBoxBlur_param * 0.1
    RBB = K.augmentation.RandomBoxBlur(p=RandomBoxBlur_param.cpu())
    out = RBB(img.cpu()).to(device)
    if torch.isnan(out).any():
        print("RandomBoxBlur_fc nan")
    return  out

def RandomEqualize(RandomEqualize_param, img):
    if torch.isnan(img).any():
        print("RandomEqualize_fc input  nan")
    if find_zero(RandomEqualize_param, RandomEqualize_param.shape[0]):
        RandomEqualize_param = deal_with_zero(RandomEqualize_param, RandomEqualize_param.shape[0]).to(
            device)
    else:
        RandomEqualize_param = RandomEqualize_param.to(device)
    RandomEqualize_param = RandomEqualize_param * 0.1
    RE = K.augmentation.RandomEqualize(p=RandomEqualize_param.cpu())
    out = RE(img.cpu()).to(device)
    if torch.isnan(out).any():
        print("RandomEqualize_fc nan")
    return out

def dilation(dilation_param, img):
    if torch.isnan(img).any():
        print("dilation input  nan")
    if find_zero(dilation_param, dilation_param.shape[0]):
        dilation_param = deal_with_zero(dilation_param, dilation_param.shape[0]).to(device)
    else:
        dilation_param = dilation_param.to(device)
    # print("param :", blur_pool2d_param )
    # print("param :",blur_pool2d_param%0.1)
    dilation_param=torch.mean(dilation_param)
    dilation_param = dilation_param * 0.5
    if dilation_param < 0.1:
        dilation_param = dilation_param + 0.1
    kernel_size = (dilation_param - (dilation_param % 0.1)).item()
    kernel_size = int(kernel_size * 10)
    # print("kernel:",kernel_size)
    kernel = torch.ones(kernel_size, kernel_size).to(device)
    # print("input shape:", input_temp.shape)
    out = K.morphology.dilation(img, kernel=kernel).to()
    if torch.isnan(out).any():
        print("dilation nan")
    return out

def erosion(erosion_param, img):
    if torch.isnan(img).any():
        print("blur_pool2d_fc input  nan")
    if find_zero(erosion_param, erosion_param.shape[0]):
        erosion_param = deal_with_zero(erosion_param, erosion_param.shape[0]).to(device)
    else:
        erosion_param = erosion_param.to(device)
    # print("param :", blur_pool2d_param )
    # print("param :",blur_pool2d_param%0.1)
    erosion_param=torch.mean(erosion_param)
    erosion_param = erosion_param * 0.5
    if erosion_param < 0.1:
        erosion_param = erosion_param + 0.1
    kernel_size = (erosion_param - (erosion_param % 0.1)).item()
    kernel_size = int(kernel_size * 10)
    kernel = torch.ones(kernel_size, kernel_size).to(device)
    # print("input shape:", input_temp.shape)
    out = K.morphology.erosion(img, kernel=kernel)
    if torch.isnan(out).any():
        print("erosion nan")
    return out

def opening(opening_param, img):
    if torch.isnan(img).any():
        print("blur_pool2d_fc input  nan")
    if find_zero(opening_param, opening_param.shape[0]):
        opening_param = deal_with_zero(opening_param, opening_param.shape[0]).to(device)
    else:
        opening_param = opening_param.to(device)
    # print("param :", blur_pool2d_param )
    # print("param :",blur_pool2d_param%0.1)
    opening_param=torch.mean(opening_param)
    opening_param = opening_param * 0.5
    if opening_param < 0.1:
        opening_param = opening_param + 0.1
    kernel_size = (opening_param - (opening_param % 0.1)).item()
    kernel_size = int(kernel_size * 10)
    kernel = torch.ones(kernel_size, kernel_size).to(device)
    out = K.morphology.opening(img, kernel=kernel)
    if torch.isnan(out).any():
        print("opening nan")
    return out

def closing(closing_param, img):
    if torch.isnan(img).any():
        print("blur_pool2d_fc input  nan")
    if find_zero(closing_param, closing_param.shape[0]):
        closing_param = deal_with_zero(closing_param, closing_param.shape[0]).to(device)
    else:
        closing_param = closing_param.to(device)
    # print("param :", blur_pool2d_param )
    # print("param :",blur_pool2d_param%0.1)
    closing_param=torch.mean(closing_param)
    closing_param = closing_param * 0.5
    if closing_param < 0.1:
        closing_param = closing_param + 0.1
    kernel_size = (closing_param - (closing_param % 0.1)).item()
    kernel_size = int(kernel_size * 10)
    kernel = torch.ones(kernel_size, kernel_size).to(device)
    out = K.morphology.closing(img, kernel=kernel)
    if torch.isnan(out).any():
        print("closing nan")
    return out



def sharpness(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")

    out = K.enhance.sharpness(input=img, factor=0.5)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out


def equalize(img):
    if torch.isnan(img).any():
        print("equalize input  nan")
    # adjust_contrast_param = adjust_contrast_param / 4 + 0.75
    out = K.enhance.equalize(img)
    if torch.isnan(out).any():
        print("equalize nan")

    return out


def box_blur(img):
    if torch.isnan(img).any():
        print("box_blur input  nan")
    out = K.filters.box_blur(input=img,kernel_size=(3,3))
    if torch.isnan(out).any():
        print("box_blur nan")

    return out

def gaussian_blur2d(gaussian_blur2d_param,img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    if find_zero(gaussian_blur2d_param, gaussian_blur2d_param.shape[0]):
        gaussian_blur2d_param = deal_with_zero(gaussian_blur2d_param,
                                               gaussian_blur2d_param.shape[0]).to(device)
    else:
        gaussian_blur2d_param = gaussian_blur2d_param.to(device)
    gaussian_blur2d_param=gaussian_blur2d_param*5
    out = K.filters.gaussian_blur2d(input=img, kernel_size=(3, 3),sigma=(gaussian_blur2d_param,gaussian_blur2d_param))
    #
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out

def sobel(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    out = K.filters.sobel(input=img)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out


def canny(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    out = K.filters.canny(input=img)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out



def rgb_to_bgr(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    out = K.color.rgb_to_bgr(img)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out



def rgb_to_hls(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    out = K.color.rgb_to_hls(img)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out

def rgb_to_hsv(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    out = K.color.rgb_to_hsv(img)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out

def rgb_to_ycbcr(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    out = K.color.rgb_to_ycbcr(img)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out


def rgb_to_yuv(img):
    if torch.isnan(img).any():
        print("adjust_contrast_fc input  nan")
    out = K.color.rgb_to_yuv(img)
    if torch.isnan(out).any():
        print("adjust_contrast_fc nan")

    return out