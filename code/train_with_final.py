from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar
import logging
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as dset
from gan.gen_with_pretrain_param_net_DiT_256_special_mask import UnetGenerator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights
import os
from tqdm import tqdm
from mv2 import mobile_net_v2
import csv
import torch
import numpy as np
from dataset import AVADataset
import option

opt = option.init()

batch_size = 8

parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="NST", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--batch_size", type=int, default=batch_size, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_image_and_score(image_tensor, score, image_name, i, j, image_folder="attack_image_dataset", csv_file="attack_image_dataset/label.csv"):
    image_name="%s/%d_%d_fake_%.6f.jpg" % (image_folder, i, j, score)

    save_image(image_tensor,image_name )
    #save_image(image_tensor, "%s" % (image_name))
    image_name = "%d_%d_fake_%.6f.jpg" % (i, j, score)

    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([image_name, score])

class M_MNet(nn.Module):
    def __init__(self, pretrained_base_model=False):
        super(M_MNet, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)

        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def get_batch_success(origin_pred, fake_pred):
    length = fake_pred.shape[0]
    worng_pred_count = 0
    for i in range(length):
        if fake_pred[i] > 5:
            if origin_pred[i] < 5:
                worng_pred_count = worng_pred_count + 1

    return worng_pred_count / length


def get_batch_if_develop(origin_pred, fake_pred):
    length = fake_pred.shape[0]
    worng_pred_count = 0
    for i in range(length):
        if fake_pred[i] > origin_pred[i]:
            worng_pred_count = worng_pred_count + 1

    return worng_pred_count / length


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def set_logging():
    now_time = time.localtime(time.time())
    # print("time:",now_time)
    filename = 'log_final_loss-3_without-if_pic-3-8__1round_16-depth-filter-choice-5conv3d_-18filter_self-init_bs-8_%s_%s_%s_%s_%s.txt' % (
        now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min, now_time.tm_sec)
    os.system(r"touch log/{}".format(filename))
    logging.basicConfig(
        level=logging.INFO,
        filename='log/' + filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
    print("filename:", filename)
    logger = logger_config(log_path=filename, logging_name='train_with_vit')
    return logger


def create_data_part(opt):
    train_csv_path = 'train.csv'
    val_csv_path = 'val.csv'
    test_csv_path = 'test.csv'

    train_ds = AVADataset(train_csv_path, opt.path_to_images, if_train=True)
    val_ds = AVADataset(val_csv_path, opt.path_to_images, if_train=False)
    test_ds = AVADataset(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader




device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms_256 = transforms.Compose([transforms.Resize(256),
                                     transforms.ToTensor(),
                                     ])

transforms_224 = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     ])


print('Defining models!')
generator = UnetGenerator(batch=batch_size).to(device)
#model_path_gen = "/runs/generator_with_final_special_mask_11_9_47.pt"
#generator.load_state_dict(torch.load(model_path_gen, map_location='cuda:0'))

discriminator = M_MNet()
model_path = 'M+MNet_vacc0.8283796740172579_srcc0.8122795303100221.pth'
discriminator.load_state_dict(torch.load(model_path, map_location='cuda:0'))
discriminator = discriminator.to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
schdeulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.1, patience=3, verbose=True,
                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                        eps=1e-08)
schdeulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.1, patience=3, verbose=True,
                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                        eps=1e-08)

g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

def get_img_score_scale_id(img_score):
    score = img_score
    if score <= 1:
        return 0
    elif score <= 2:
        return 1
    elif score <= 3:
        return 2
    elif score <= 4:
        return 3
    elif score <= 5:
        return 4
    elif score <= 6:
        return 5
    elif score <= 7:
        return 6
    elif score <= 8:
        return 7
    elif score <= 9:
        return 8
    elif score <= 10:
        return 9


def get_img_score_scale(id):
    if id == 1:
        return "0-1"
    elif id == 2:
        return "1-2"
    elif id == 3:
        return "2-3"
    elif id == 4:
        return "3-4"
    elif id == 5:
        return "4-5"
    elif id == 6:
        return "5-6"
    elif id == 7:
        return "6-7"
    elif id == 8:
        return "7-8"
    elif id == 9:
        return "8-9"
    elif id == 10:
        return "9-10"


def get_scale_mean_development(origin_score, fake_score):
    count_scale = torch.zeros(10)
    count_scale_pred = torch.zeros(10)
    development_scale = torch.zeros(10)
    for i in range(batch_size):
        count_scale[get_img_score_scale_id(origin_score[i]) - 1] = count_scale[
                                                                       get_img_score_scale_id(origin_score[i] - 1)] + 1
        count_scale_pred[get_img_score_scale_id(fake_score[i]) - 1] = count_scale_pred[
                                                                          get_img_score_scale_id(fake_score[i] - 1)] + 1
        development_scale[get_img_score_scale_id(origin_score[i]) - 1] = development_scale[get_img_score_scale_id(
            origin_score[i] - 1)] + (fake_score[i] - origin_score[i])
    return count_scale, development_scale, count_scale_pred


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


def get_filter_param_scale(id):
    if id == 0:
        return "(0, 2)"
    if id == 1:
        return "(-1,1) "
    if id == 2:
        return "(0, 2) "
    if id == 3:
        return "(0, 2)"
    if id == 4:
        return "(-1,1) "
    if id == 5:
        return "(0, 2)"
    if id == 6:
        return "(1,2,3...9)"
    if id == 7:
        return "(1,2,3,4,5)"
    if id == 8:
        return "(1,2,3,4,5)"
    if id == 9:
        return "(1,2,3,4,5) "
    if id == 10:
        return "(1,2,3,4,5)"
    if id == 11:
        return "none param"
    if id == 12:
        return "none param"
    if id == 13:
        return "none param"
    if id == 14:
        return "none param"
    if id == 15:
        return "none param"
    if id == 16:
        return "none param"
    if id == 17:
        return "none param"


def get_score(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score


def get_batch_acc(label, origin_pred):
    length = label.shape[0]
    flag = 0
    for i in range(length):
        if label[i] >= 5:
            if origin_pred[i] >= 5:
                flag = flag + 1
        elif label[i] < 5:
            if origin_pred[i] < 5:
                flag = flag + 1

    return flag / length


logger_model = Logger(filename=args.dataset)
print('Start of training process!')
logger = set_logging()


train_loader, val_loader, test_loader = create_data_part(opt)

for epoch in range(args.epochs):

    discriminator.eval()
    generator.train()

    ge_loss = 0.
    de_loss = 0.
    Generator_score = 0.
    full_socre = torch.ones(batch_size) * 10
    Discriminator_right_score = 0.
    Discriminator_false_score = 0.
    img_score_development = 0.
    success = 0.
    if_develop = 0.
    batch_acc = 0.

    count_scale_all = torch.zeros(10)
    count_scale_all_pred = torch.zeros(10)
    scale_develop_mean_all = torch.zeros(10)
    start = time.time()
    temp_fake = torch.zeros([3, 56, 256]).to(device)
    bar = IncrementalBar(f'[Epoch {epoch + 1}/{args.epochs}]', max=len(train_loader))
    all_fake_result_temp = torch.zeros(16, 2)

    for i, data in enumerate(tqdm(train_loader)):
        x = data[0].to(device)

        fake, record, filter_weight = generator(x, epoch=epoch, small_epoch=i, times=i)

        fake_pred = get_score(discriminator(fake))


        origin_pred = get_score(discriminator(x))
        origin_pred_for_loss = origin_pred+5
        g_loss = g_criterion(x,fake, origin_pred_for_loss, fake_pred)
        g_loss.backward()
        g_optimizer.step()

        biggest_improve = torch.max(fake_pred - origin_pred)
        ge_loss += g_loss.item()
        img_score_development += (fake_pred - origin_pred).mean()
        count_temp, develop_temp, count_fake_temp = get_scale_mean_development(origin_pred, fake_pred)
        count_scale_all = count_scale_all + count_temp
        count_scale_all_pred = count_scale_all_pred + count_fake_temp
        scale_develop_mean_all = scale_develop_mean_all + develop_temp

        success = success + get_batch_success(origin_pred, fake_pred)
        if_develop = if_develop + get_batch_if_develop(origin_pred, fake_pred)


        bar.next()

        if i % 512 == 0:
            logger.info(" ")
            logger.info("____________________________________________________________________________________________")
            logger.info("____________________________________________________________________________________________")

            fake_score = discriminator(fake[3].reshape(1, 3, 256, 256))
            fake_score = get_score(fake_score)
            origin_score = get_score(discriminator(x[3].reshape(1, 3, 256, 256)))
            count_scale = count_scale_all
            count_pred = count_scale_all_pred
            scale_develop = scale_develop_mean_all
            for j in range(10):
                if count_scale[j] != 0:
                    scale_develop[j] = scale_develop[j] / count_scale[j]
            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info("this is the mean development on different socre scale:")
            for j in range(10):
                logger.info(
                    "scale %-7s origin have  %-10d  pics ,after pred have %-10d  ,while the mean development is %-8.6f" % (
                    get_img_score_scale(j + 1), count_scale[j], count_pred[j], scale_develop[j]))
            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info("this the discrimeinator output score of the saved  fake pic[3]:")
            logger.info("score [%.4f]:" % fake_score[0])
            logger.info("this the discrimeinator output score of the saved  original pic[3]:")
            logger.info("score [%.4f]:" % origin_score[0])
            logger.info("Generator     loss:::::::%.6f" % (ge_loss / (i + 1)))
            logger.info("the mean development of the original pic    :::::::%.6f" % (img_score_development / 512))
            img_score_development = 0.
            logger.info("the biggest development of the original pic :::::::%.6f" % (biggest_improve))
            logger.info("how many pic have been developed to >5      :::::::%.6f" % (success / 512))
            logger.info("how many pic have been developed            :::::::%.6f" % (if_develop / 512))

            success = 0.
            if_develop = 0.
            batch_acc = 0.

            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info("this is the mean param and weight of different filters:")
            for k in range(14):
                logger.info(
                    "the filter %-20s 's mean param is [%-7.6f] with param scale of :%-12s  and the weight is [%-7.6f]  " % (
                    trans_id_to_filter(k), record[k][3], get_filter_param_scale(k), filter_weight[3][k]))

            logger.info("____________________________________________________________________________________________")

    bar.finish()

    g_loss = ge_loss / len(dataloader_lowpoint)
    d_loss = de_loss / len(dataloader_lowpoint)

    end = time.time()
    tm = (end - start)
    logger.info('epoch %d:   generator_loss       %.4f' % (epoch + 1, g_loss))
    logger.info('epoch %d:   discriminator_loss   %.4f' % (epoch + 1, d_loss))
    now_time = time.localtime(time.time())
    logger_model.save_weights(generator.state_dict(),
                              'generator_with_final_plane_%s_%s_%s' % (
                              now_time.tm_hour, now_time.tm_min, now_time.tm_sec))

    logger.info("[Epoch %d/%d] [G loss(0.01): %.3f] [D loss(0.01): %.3f] ETA: %.3fs" % (
        epoch + 1, args.epochs, g_loss * 100, d_loss * 100, tm))
logger_model.close()
print('End of training process!')