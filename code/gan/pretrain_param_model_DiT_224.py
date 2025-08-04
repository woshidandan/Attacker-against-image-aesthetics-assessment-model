from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb
from torch import nn
from .DiT.DiT_model_param_saturation_224 import DiffusionModel_Saturation
from .DiT.DiT_model_param_brightness_224 import DiffusionModel_Bright
from .DiT.DiT_model_param_contrast_224 import DiffusionModel_Contrast
from .DiT.DiT_model_param_gamma_224 import DiffusionModel_Gamma
from .DiT.DiT_model_param_hue_224 import DiffusionModel_Hue
import torch



main_path="gan/DiT/saved_model/"

"""
#单个美学因素的CNN decoder网络
class Saturation_Decoder(nn.Module):
    def __init__(self):
        super(Saturation_Decoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(32, , kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 1, kernel_size=3, stride=16, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.squeeze(x.view(x.size(0), -1))
        x= 2*x+0.01
        return x

class Brightness_Decoder(nn.Module):
    def __init__(self):
        super(Brightness_Decoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(32, , kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 1, kernel_size=3, stride=16, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        #print("x shape:",x.shape)

        #x = torch.squeeze(self.fc_layer(x))
        x = torch.squeeze(x.view(x.size(0), -1))
        #print("x shape:", x.shape)
        x=(x-0.5)*2
        #x=x*0.4

        return x

class Contrast_Decoder(nn.Module):
    def __init__(self):
        super(Contrast_Decoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(32, , kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 1, kernel_size=3, stride=16, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        #print("x shape:",x.shape)
        #x = torch.squeeze(self.fc_layer(x))
        x = torch.squeeze(x.view(x.size(0), -1))
        #print("x shape:", x.shape)
        #x=(x-0.5)*2
        #x=x*0.4
        #[0, 1]->[0.5, 2]  [0.667, 2]
        x= 2*x+0.01
        return x

class Gamma_Decoder(nn.Module):
    def __init__(self):
        super(Gamma_Decoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(32, , kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 1, kernel_size=3, stride=16, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.squeeze(x.view(x.size(0), -1))
        x=x*2 #[0, 2]
        #x= x
        return x

class Hue_Decoder(nn.Module):
    def __init__(self):
        super(Hue_Decoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(32, , kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 1, kernel_size=3, stride=16, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.squeeze(x.view(x.size(0), -1))
        x=(x-0.5)*2
        x= pi*x #p[-pi, pi]
        return x

"""

#由多个decoder组成的diffusion网络部分
class Saturation_Param_head(nn.Module):
    def __init__(self, num_encoders=3, num_decoders=3, dix=0):
        super().__init__()
        #self.num_encoders = num_encoders
        #self.num_decoders = num_decoders
        #self.decoder1 = Saturation_Decoder()
        #self.decoder2 = Saturation_Decoder()
        #self.decoder3 = Saturation_Decoder()
        self.DiT_saturation=DiffusionModel_Saturation()

    def forward(self, x):
        #image = x
        #decode_param1 = self.decoder1(image)
        #image = adjust_saturation(decode_param1, image)
        #decode_param2 = self.decoder2(image)
        #image = adjust_saturation(decode_param2, image)
        #decode_param3 = self.decoder3(image)
        #image = adjust_saturation(decode_param3, image)

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


#加在diffusion部分之后的整形网络，根据输出张量的channel=3和=1，分为了两种
class Tail_ConvNet(nn.Module):
    def __init__(self):
        super(Tail_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        #print("x shape:",x.shape)
        x = nn.functional.relu(self.fc1(x))
        #print("x shape:", x.shape)
        x = nn.functional.sigmoid(self.fc2(x))
        #stop
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
        self.fc1 = nn.Linear(256 * 7 *7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # print("1",x.shape)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # print("2",x.shape)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # print("3",x.shape)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        # print("4",x.shape)
        x = self.pool(nn.functional.relu(self.conv4(x)))
        # print("5",x.shape)
        x = self.pool(nn.functional.relu(self.conv5(x)))
        # print("6",x.shape)
        x = x.view(x.size(0), -1)
        # print("7",x.shape)
        x = nn.functional.relu(self.fc1(x))
        # print("8",x.shape)
        x = nn.functional.sigmoid(self.fc2(x))
        # print("9",x.shape)
        # stop
        return x



#组合diffusion部分和整形网络部分，并读取预训练的pth文件
class Saturation_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net=DiffusionModel_Saturation()
        self.tail_net=Tail_ConvNet()
        #model_path = '/home/hzy/attack_frame/train_with_temp/gan/DiT/saturation_batch_8_224_epoch_0_loss0.008888006313766936.pth'
        #self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        #for name, parameter in self.param_net.named_parameters():
        #    parameter.requires_grad = False

    def forward(self, x,times):
        param=self.tail_net(self.param_net(x,times))

        return param

class Brightness_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Bright()
        self.tail_net = Tail_ConvNet()
        #model_path = '/home/hzy/attack_frame/train_with_temp/gan/DiT/brightness_batch_8_224_epoch_0_loss0.02897613557462645.pth'
        #self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        #for name, parameter in self.param_net.named_parameters():
        #    parameter.requires_grad = False

    def forward(self, x,times):
        param = self.tail_net(self.param_net(x,times))

        return param

class Contrast_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Contrast()
        self.tail_net = Tail_ConvNet()
        #model_path = '/home/hzy/attack_frame/train_with_temp/gan/DiT/contrast_batch_8_224_epoch_0_loss0.028845393863718306.pth'
        #self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        #for name, parameter in self.param_net.named_parameters():
        #    parameter.requires_grad = False

    def forward(self, x,times):
        deep_img=self.param_net(x,times)
        #print("deep_img shape:",deep_img.shape)
        #print("+++++++++++++++++++++++++++++++++++++++++")
        param = self.tail_net(deep_img)
        #print("contrast param shape:", param.shape)

        return param

class Gamma_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Gamma()
        self.tail_net = Tail_ConvNet()
        #model_path = '/home/hzy/attack_frame/train_with_temp/gan/DiT/gamma_batch_8_224_epoch_0_loss0.02769880125336419.pth'
        #self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        #for name, parameter in self.param_net.named_parameters():
        #    parameter.requires_grad = False

    def forward(self, x,times):
        param = self.tail_net(self.param_net(x,times))

        return param

class Hue_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = DiffusionModel_Hue()
        self.tail_net = Tail_ConvNet()
        #model_path = '/home/hzy/attack_frame/train_with_temp/gan/DiT/hue_batch_8_224_epoch_0_loss0.03233133521252916.pth'
        #self.param_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        #for name, parameter in self.param_net.named_parameters():
        #    parameter.requires_grad = False

    def forward(self, x,times):
        #print("x shape:",x.shape)
        h, s, v = torch.chunk(rgb_to_hsv(x), chunks=3, dim=-3)
        #print("h shape:",h.shape)
        h=h.repeat(1, 3, 1, 1)

        #print("h shape:",h.shape)
        #h = h.to(device)
        #print("h shape:",h.shape)
        param = self.tail_net(self.param_net(h,times))

        return param