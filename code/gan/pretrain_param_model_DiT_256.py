from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb
from torch import nn
import torch
from .DiT.DiT_model_param_saturation_256 import DiffusionModel_Saturation
from .DiT.DiT_model_param_brightness_256 import DiffusionModel_Bright
from .DiT.DiT_model_param_contrast_256 import DiffusionModel_Contrast
from .DiT.DiT_model_param_gamma_256 import DiffusionModel_Gamma
from .DiT.DiT_model_param_hue_256 import DiffusionModel_Hue




class Saturation_Param_head(nn.Module):
    def __init__(self, num_encoders=3, num_decoders=3, dix=0):
        super().__init__()
        self.DiT_saturation=DiffusionModel_Saturation()

    def forward(self, x):

        image=self.DiT_saturation(x)

        return image

class Brightness_Param_head(nn.Module):

    def __init__(self, num_encoders=3, num_decoders=3,dix=0):
        super().__init__()

        self.DiT_brightness=DiffusionModel_Bright()

    def forward(self, x):
        image = self.DiT_brightness(x)

        return image

class Contrast_Param_head(nn.Module):
    def __init__(self, num_encoders=3, num_decoders=3, dix=0):
        super().__init__()
        self.DiT_contrast=DiffusionModel_Contrast()

    def forward(self, x):
        image = self.DiT_contrast(x)

        return image

class Gamma_Param_head(nn.Module):
    def __init__(self, num_encoders=3, num_decoders=3, dix=0):
        super().__init__()
        self.DiT_gamma=DiffusionModel_Gamma()

    def forward(self, x):
        image = self.DiT_gamma(x)

        return image

class Hue_Param_head(nn.Module):
    def __init__(self, num_encoders=3, num_decoders=3, dix=0):
        super().__init__()
        self.DiT_hue=DiffusionModel_Hue

    def forward(self, x):
        image = self.DiT_hue(x)

        return image


class Tail_ConvNet(nn.Module):
    def __init__(self):
        super(Tail_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x

class Tail_ConvNet_one_dim_input(nn.Module):
    def __init__(self):
        super(Tail_ConvNet_one_dim_input, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 8 *8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x



class Saturation_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net=DiffusionModel_Saturation()
        self.tail_net=Tail_ConvNet()
        model_path = 'gan/DiT/saturation_batch_8_256_epoch_0_loss0.007913792306671307.pth'
        self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    def forward(self, x,times):
        param=self.tail_net(self.param_net(x,times))

        return param

class Brightness_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Bright()
        self.tail_net = Tail_ConvNet()
        model_path = 'gan/DiT/brightness_batch_8_256_epoch_0_loss0.027612949186456653.pth'
        self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    def forward(self, x,times):
        param = self.tail_net(self.param_net(x,times))

        return param

class Contrast_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Contrast()
        self.tail_net = Tail_ConvNet()
        model_path = 'gan/DiT/contrast_batch_8_256_epoch_0_loss0.02664747554692221.pth'
        self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    def forward(self, x,times):
        deep_img=self.param_net(x,times)
        param = self.tail_net(deep_img)

        return param

class Gamma_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Gamma()
        self.tail_net = Tail_ConvNet()
        model_path = 'gan/DiT/gamma_batch_8_256_epoch_0_loss0.02757305865751611.pth'
        self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    def forward(self, x,times):
        param = self.tail_net(self.param_net(x,times))

        return param

class Hue_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Hue()
        self.tail_net = Tail_ConvNet_one_dim_input()
        model_path = 'gan/DiT/hue_batch_8_256_epoch_0_loss0.006658911409167632.pth'
        self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    def forward(self, x,times):
        h, s, v = torch.chunk(rgb_to_hsv(x), chunks=3, dim=-3)
        param = self.tail_net(self.param_net(h,times))

        return param