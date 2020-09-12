import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import cv2
import time

from unet.unet_model import *
from predict_brats import predict
from utils.dataset import BrainDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dice_loss import dice_coeff
from utils.init_logging import init_logging

dir_checkpoint = 'checkpoints/'
train_list = 'data/train_brats_only_tumor.txt'
val_list = 'data/val_brats.txt'
test_list = 'data/test_brats.txt'

def train_net(reconstucter,
              segmenter,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    train_set = BrainDataset(train_list,has_mean=True) #
    val_set = BrainDataset(val_list)    #test
    test_set = BrainDataset(val_list,test_list)
    n_val = len(val_set)
    n_train = len(train_set)
    n_test = len(test_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    logger.info(f'''Starting training:
        Starting time    {starttime_r}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        λ:               {lambd}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer_R = optim.Adam(reconstucter.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer_S = optim.Adam(segmenter.parameters(), lr=lr, betas=(0.9, 0.99))

    scheduler_R = optim.lr_scheduler.ReduceLROnPlateau(optimizer_R, 'min' if reconstucter.n_classes > 1 else 'max', patience=2)
    scheduler_S = optim.lr_scheduler.ReduceLROnPlateau(optimizer_S, 'min' if segmenter.n_classes > 1 else 'max', patience=2)

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss(reduction='none')
    sigmoid = nn.Sigmoid()
    best_dice = 0
    for epoch in range(epochs):
        reconstucter.train()
        segmenter.train()
        loss1 = 0
        loss2 = 0
        loss3 = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img',ncols=120,ascii=True) as pbar:
            for i,(img, seg, seg_tumor, [name,slice_idx], [mean,std]) in enumerate(train_loader):
                batch, h, w = img.shape[0], img.shape[-2], img.shape[-1]
                img = Variable(img.unsqueeze(dim=1).to(device=device, dtype=torch.float32))
                seg = seg.to(device=device, dtype=torch.long)
                seg_tumor = seg_tumor.to(device=device, dtype=torch.long)
                seg_no_tumor = torch.zeros_like(seg).to(device=device, dtype=torch.long)
                # seg_tumor = seg_tumor.to(device=device, dtype=torch.float32)
                
                #train step 1:
                optimizer_S.zero_grad()
                img_pred = reconstucter(img)
                seg_pred = segmenter(img_pred)

                loss_seger = lambd[0]*ce_loss(seg_pred,seg_tumor)
                loss_seger.backward()
                optimizer_S.step()

                #train step 2:
                optimizer_R.zero_grad()
                img_pred = reconstucter(img)
                seg_pred = segmenter(img_pred)

                loss_dice = lambd[1]*ce_loss(seg_pred,seg_no_tumor)
                seg_tumor_neg = torch.where(seg_tumor==1,torch.zeros_like(seg),torch.ones_like(seg))
                loss_mse = lambd[2]*(mse_loss(img_pred,img)).mean() #直接返回img_pred大小的loss向量，乘上mask，只计算肿瘤区域以外的loss
                loss_recer = loss_dice + loss_mse
                loss_recer.backward()
                optimizer_R.step()       

                pbar.set_postfix_str('loss(batch):{:>.2e},{:>.2e},{:>.2e}'.format(loss_seger.item(),loss_dice.item(),loss_mse.item()))
                loss1 += loss_seger
                loss2 += loss_dice
                loss3 += loss_mse
                pbar.update(batch)
                # break
        logger.info('Epoch loss: {:.2e}, {:.2e}, {:.2e}'.format(loss1/i,loss2/i,loss3/i))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logger.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save({'reconstucter':reconstucter.state_dict(),
                        'segmenter':segmenter.state_dict()},
                       dir_checkpoint + f'{starttime}.pth')
            logger.info(f'Checkpoint {epoch + 1} saved !')
        # break
    predict(args,
            starttime=starttime,
            reconstucter=reconstucter,
            segmenter=segmenter,
            test_loader=test_loader,
            logger=logger)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-g', '--gpu', dest='gpu', default=3, type=int, help='witch gpu to use')
    parser.add_argument('--output', '-o', metavar='OUTPUT',type=str,default='test_2020-08-19_12-23-21',
                             help='Filenames of ouput images', dest='output')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',type=str,
                             help='filenames of input images')
    parser.add_argument('--model', '-m', type=str,default='checkpoints/2020-08-19_12-23-21.pth',
                             metavar='FILE',help="Specify the file in which the model is stored")
    parser.add_argument('--lambd', '-d', type=str,default='[1,1,1]',dest='lambd',
                             metavar='FILE',help="param of 3 losses")

    return parser.parse_args()


if __name__ == '__main__':
    t = time.localtime()
    starttime = time.strftime("%Y-%m-%d_%H-%M-%S", t)
    starttime_r = time.strftime("%Y/%m/%d %H:%M:%S", t) #readable time

    logger = init_logging(starttime)
    args = get_args()
    lambd = eval(args.lambd)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True # faster convolutions, but more memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    # Change here to adapt to your data 
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    reconstucter = Reconstucter(n_channels=1, n_classes=1, bilinear=True)
    segmenter = Segmenter(n_channels=1, n_classes=2, bilinear=True)
    logger.info(f'Network:\n'
                 f'\t{reconstucter.n_channels} input channels\n'
                 f'\t{segmenter.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if segmenter.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logger.info(f'Model loaded from {args.load}')

    reconstucter.to(device=device)
    segmenter.to(device=device)

    try:
        train_net(reconstucter=reconstucter,
                  segmenter=segmenter,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
        logger.info('Program done. Start on %s'%(starttime_r))
    except KeyboardInterrupt:
        torch.save({'reconstucter':reconstucter.state_dict(),
                        'segmenter':segmenter.state_dict()}, './checkpoints/INTERRUPTED.pth')
        logger.info('User canceled, start on %s, Saved interrupt.'%(starttime_r))
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
        try:
            os.remove(f'./log/{starttime}.txt')
        except:
            pass
    except:
        logger.info('Error! Start on %s'%(starttime_r))
