import torch
import torch.nn.functional as F
from .filter_functions import *
from .pretrain_param_model_DiT_224 import *
from torchvision.utils import save_image
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


def trans_id_to_filter(id):
    if id == 0:
        return "adjust_contrast"
    if id == 1:
        return "adjust_brightness"
    if id == 2:
        return "adjust_saturation"
    if id == 3:
        return "adjust_gamma"
    if id == 4:
        return "adjust_hue"
    if id == 5:
        return "sharpness"
    if id == 6:
        return "box_blur"
    # if id == 7:
    #    return "dilation"
    # if id == 8:
    #    return "erosion"
    # if id == 9:
    #    return "opening"
    # if id == 10:
    #    return "closing"
    if id == 7:
        return "equalize"
    if id == 8:
        return "sobel"
    if id == 9:
        return "rgb_to_yuv"
    if id == 10:
        return "rgb_to_bgr"
    if id == 11:
        return "rgb_to_hls"
    if id == 12:
        return "rgb_to_hsv"
    if id == 13:
        return "rgb_to_ycbcr"


class MultiFrameTensorWeightNet(nn.Module):
    def __init__(self, input_channels, num_frames, hidden_size, output_size):
        super(MultiFrameTensorWeightNet, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(num_frames, 2, 2), stride=2,
                               padding=(num_frames // 2, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 512, kernel_size=(2, 2, 2), stride=2, padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(512)
        self.dropout2 = nn.Dropout3d(p=0.5)
        self.conv3 = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(1024)
        self.conv4 = nn.Conv3d(1024, 512, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.conv5 = nn.Conv3d(512, 256, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(256)
        self.conv6 = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(128)
        self.conv7 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=4, padding=(1, 1, 1))
        self.bn7 = nn.BatchNorm3d(64)
        self.dropout7 = nn.Dropout3d(p=0.5)
        self.fc1 = nn.Linear(64, 14)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(1, 2, 0, 3, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout7(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # print("x shape:",x.shape)

        x = torch.softmax(x, dim=1)
        return x


class Mask_ConvNet(nn.Module):
    def __init__(self, degree):
        super(Mask_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.degree=degree

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        k = int(self.degree * x.numel())
        x_temp = torch.flatten(x)
        _, indices = torch.topk(x_temp, k)
        threshold = torch.min(x_temp[indices])
        output = torch.where(x >= threshold, 1, 0)

        return output


class Mask_ConvNet_one_dim(nn.Module):
    def __init__(self):
        super(Mask_ConvNet_one_dim, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(256 * 8 * 8, 128)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        # x = self.pool(nn.functional.relu(self.conv5(x)))
        # x = x.view(x.size(0), -1)
        # x = nn.functional.relu(self.fc1(x))
        # x = nn.functional.sigmoid(self.fc2(x))
        # x=nn.functional.sigmoid(x)

        # 计算要转换的元素的数量
        k = int(0.02 * x.numel())
        # 获取最大的k元素的索引
        x_temp = torch.flatten(x)
        # print("x_temp ")
        _, indices = torch.topk(x_temp, k)
        threshold = torch.min(x_temp[indices])
        # 将最大的k个元素设为1，其余元素设为0
        output = torch.where(x >= threshold, 1, 0)

        return output

class UnetGenerator(nn.Module):
    def __init__(self, drop=0.1, batch=0):
        super().__init__()
        self.filter_num = 14
        self.batch = batch
        self.paces = 1

        self.fliter_type_one_pace_mul = MultiFrameTensorWeightNet(input_channels=3, num_frames=14, hidden_size=256,
                                                                  output_size=14)

        self.adjust_brightness_fc = Brightness_Param()
        self.adjust_contrast_fc = Contrast_Param()
        self.adjust_saturation_fc = Saturation_Param()
        self.adjust_gamma_fc = Gamma_Param()
        self.adjust_hue_fc = Hue_Param()


        self.mask_brightness_fc = Mask_ConvNet(degree=0.6)
        self.mask_contrast_fc = Mask_ConvNet(degree=0.6)
        self.mask_saturation_fc = Mask_ConvNet(degree=0.6)
        self.mask_gamma_fc = Mask_ConvNet(degree=0.6)
        self.mask_hue_fc = Mask_ConvNet(degree=0.6)
        self.mask_sharpness_fc = Mask_ConvNet(degree=0.6)
        self.mask_box_blur_fc = Mask_ConvNet(degree=0.6)
        self.mask_equalize_fc = Mask_ConvNet(degree=0.6)
        self.mask_sobel_fc = Mask_ConvNet(degree=0.1)
        self.mask_rgb_to_yuv_fc = Mask_ConvNet(degree=0.01)
        self.mask_rgb_to_bgr = Mask_ConvNet(degree=0.01)
        self.mask_rgb_to_hls = Mask_ConvNet(degree=0.01)
        self.mask_rgb_to_hsv_fc = Mask_ConvNet(degree=0.01)
        self.mask_rgb_to_ycbcr_fc = Mask_ConvNet(degree=0.01)



    def forward(self, x, epoch, small_epoch, times):
        img = x
        # save_image(img[3], "%s/origin.jpg" % ("g_images_sequence"))
        length = img.shape[0]
        filter_param_record = torch.zeros([self.paces, length, 2])
        #print(torch.min(img), torch.max(img))
        #print("x device:",img.device)
        #print("self.adjust_contrast_fc device:",next(self.adjust_contrast_fc.parameters()).is_cuda)

        #img = img / 2 + 0.5
        d1 = img.to(device)
        result = torch.zeros(self.filter_num, length, 3, 224, 224).to(device)
        output = torch.zeros(img.shape).to(device)
        record = torch.zeros(self.filter_num, length).to(device)


        adjust_contrast_param = self.adjust_contrast_fc(d1,times)
        adjust_brightness_param = self.adjust_brightness_fc(d1,times)
        adjust_saturation_param = self.adjust_saturation_fc(d1,times)
        adjust_gamma_param = self.adjust_gamma_fc(d1,times)
        adjust_gamma_param=torch.ones(adjust_gamma_param.shape)*0.99999
        adjust_gamma_param=adjust_gamma_param.to(device)
        adjust_hue_param = self.adjust_hue_fc(d1,times)




        record[0] = record[0] + adjust_contrast_param[:, 0]
        record[1] = record[1] + adjust_brightness_param[:, 0]
        record[2] = record[2] + adjust_saturation_param[:, 0]
        record[3] = record[3] + adjust_gamma_param[:, 0]
        record[4] = record[4] + adjust_hue_param[:, 0]

        #print(adjust_contrast(torch.squeeze(adjust_contrast_param), img).shape)
        img=img.to(torch.float64)
        img=img.to(device)
        #print(torch.min(img),torch.max(img))
        #stop
        #print("img type:",img.type())
        #stop


        all_ones=torch.ones(d1.shape)

        result[0] = result[0] + d1*(1-self.mask_contrast_fc(adjust_contrast(torch.squeeze(adjust_contrast_param), img).float())) + adjust_contrast(torch.squeeze(adjust_contrast_param), img)*self.mask_contrast_fc(adjust_contrast(torch.squeeze(adjust_contrast_param), img).float())
        result[1] = result[1] + d1*(1-self.mask_brightness_fc(adjust_brightness(torch.squeeze(adjust_brightness_param), img).float()))+adjust_brightness(torch.squeeze(adjust_brightness_param), img)*self.mask_brightness_fc(adjust_brightness(torch.squeeze(adjust_brightness_param), img).float())
        result[2] = result[2] + d1*(1-self.mask_saturation_fc(adjust_saturation(torch.squeeze(adjust_saturation_param), img).float()))+adjust_saturation(torch.squeeze(adjust_saturation_param), img)*self.mask_saturation_fc(adjust_saturation(torch.squeeze(adjust_saturation_param), img).float())
        result[3] = result[3] + d1*(1-self.mask_gamma_fc(adjust_gamma(torch.squeeze(adjust_gamma_param), img+0.0001).float()))+adjust_gamma(torch.squeeze(adjust_gamma_param), img+0.0001)*self.mask_gamma_fc(adjust_gamma(torch.squeeze(adjust_gamma_param), img+0.0001).float())
        result[4] = result[4] + d1*(1-self.mask_hue_fc(adjust_hue(torch.squeeze(adjust_hue_param), img).float()))+adjust_hue(torch.squeeze(adjust_hue_param), img)*self.mask_hue_fc(adjust_hue(torch.squeeze(adjust_hue_param), img).float())

        result[5] = result[5] + d1*(1-self.mask_sharpness_fc(sharpness(img).float()))+sharpness(img)*self.mask_sharpness_fc(sharpness(img).float())
        result[6] = result[6] + d1*(1-self.mask_box_blur_fc(box_blur(img).float()))+box_blur(img)*self.mask_box_blur_fc(box_blur(img).float())

        result[7] = result[7] + d1*(1-self.mask_equalize_fc(equalize(img).float()))+equalize(img)*self.mask_equalize_fc(equalize(img).float())
        result[8] = result[8] + d1#*(1-self.mask_sobel_fc(sobel(img).float()))+sobel(img)*self.mask_sobel_fc(sobel(img).float())
        result[9] = result[9] + d1*(1-self.mask_rgb_to_yuv_fc(rgb_to_yuv(img).float()))+rgb_to_yuv(img)*self.mask_rgb_to_yuv_fc(rgb_to_yuv(img).float())
        result[10] = result[10] + d1*(1-self.mask_rgb_to_bgr(rgb_to_bgr(img).float()))+rgb_to_bgr(img)*self.mask_rgb_to_bgr(rgb_to_bgr(img).float())
        result[11] = result[11] + d1*(1-self.mask_rgb_to_hls(rgb_to_hls(img).float()))+rgb_to_hls(img)*self.mask_rgb_to_hls(rgb_to_hls(img).float())
        result[12] = result[12] + d1*(1-self.mask_rgb_to_hsv_fc(rgb_to_hsv(img).float()))+rgb_to_hsv(img)*self.mask_rgb_to_hsv_fc(rgb_to_hsv(img).float())
        result[13] = result[13] + d1*(1-self.mask_rgb_to_ycbcr_fc(rgb_to_ycbcr(img).float()))+rgb_to_ycbcr(img)*self.mask_rgb_to_ycbcr_fc(rgb_to_ycbcr(img).float())
        """

        result[0] = result[0] + adjust_contrast(torch.squeeze(adjust_contrast_param), img)
        result[1] = result[1] + adjust_brightness(torch.squeeze(adjust_brightness_param), img)
        result[2] = result[2] + adjust_saturation(torch.squeeze(adjust_saturation_param), img)
        result[3] = result[3] + adjust_gamma(torch.squeeze(adjust_gamma_param), img)
        result[4] = result[4] + adjust_hue(torch.squeeze(adjust_hue_param), img)

        result[5] = result[5] + sharpness(img)
        result[6] = result[6] + box_blur(img)

        result[7] = result[7] + equalize(img)
        result[8] = result[8] + sobel(img)
        result[9] = result[9] + rgb_to_yuv(img)
        result[10] = result[10] + rgb_to_bgr(img)
        result[11] = result[11] + rgb_to_hls(img)
        result[12] = result[12] + rgb_to_hsv(img)
        result[13] = result[13] + rgb_to_ycbcr(img)
        """

        weight = self.fliter_type_one_pace_mul(result)

        for i in range(length):
            for j in range(self.filter_num):
                output[i] = output[i] + result[j, i, :, :, :] * weight[i, j]

        #output = (output - 0.5) * 2
        output = torch.clamp(output, 0, 1)

        #if small_epoch % 512 == 0:
        #    save_image(img[3], "%s/origin_%d.jpg" % ("g_images_interpert",small_epoch))
        #    save_image(output[3], "%s/fake_%d.jpg" % ("g_images_interpert",small_epoch))
            #for j in range(self.filter_num):
            #    save_image(result[j][3], "%s/filter_%s.jpg" % ("g_images_interpert", trans_id_to_filter(j)))

            #save_image(output[3], "%s/fake_%f.jpg" % ("g_images_sequence",record[5][3]))

        # save_image(output[3], "%s/fake_test.jpg" % ("g_images_sequence"))

        return output, record, weight