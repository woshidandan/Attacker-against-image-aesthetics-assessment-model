import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.loss=nn.MSELoss()

    def forward(self,x,fake,fake_target,fake_pred):
        loss_1=self.loss(x,fake)
        loss = self.loss(fake_target, fake_pred)
        #print("loss:",loss)
        #stop
        return loss

class loss_both_score_image(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.loss=nn.MSELoss()

    def forward(self, fake_target,fake_pred,origin_img, fake_img):
        loss_1 = self.loss(fake_target, fake_pred)
        loss_2=self.l1(origin_img, fake_img)
        loss=0.5*loss_1+0.5*loss_2

        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss = nn.MSELoss()

    def forward(self, gt_label,origin_pred, fake_target, fake_pred):

        loss = 0.5*self.loss(fake_target, fake_pred)+0.5*self.loss(gt_label,origin_pred)


        return loss