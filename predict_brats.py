import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms
from utils.dataset import BrainDataset
from torch.utils.data import DataLoader

from unet.unet_model import *
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils.ms_ssim import MS_SSIM
from dice_loss import dice_coeff
from tqdm import tqdm
import prettytable as pt
from utils.init_logging import init_logging
import time
# from utils.Evalution_Vessel import Evalution_Vessel

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str,default='checkpoints/2020-08-19_12-23-21.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batch_size')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',type=str,
                        help='filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT',type=str,default='test_2020-08-19_12-23-21',
                        help='Filenames of ouput images', dest='output')
    parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int, help='witch gpu to use')
    
    return parser.parse_args()

def predict(args,
            logger=None,
            starttime=None,
            reconstucter=None,
            segmenter=None,
            test_loader=None):
    
    #读入val和test的case名称
    with open('data/test_brats_list.txt','r') as fp:
        test_case_list = fp.readlines()
    for i in range(len(test_case_list)):
        test_case_list[i] = test_case_list[i].strip()
    with open('data/val_brats_list.txt','r') as fp:
        val_case_list = fp.readlines()
    for i in range(len(val_case_list)):
        val_case_list[i] = val_case_list[i].strip()

    in_files = args.input
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if starttime != None:
        model_pth = 'checkpoints/%s.pth'%(starttime)
        output = '/home/lx/unet_gan_data/test_%s'%(starttime)
    else:
        model_pth = args.model
        output = args.ouput

    output_img = output+'/img'
    output_npy = output+'/npy'
    output_eval = output+'/eval'
    if not os.path.exists(output):
        os.mkdir(output)
    if not os.path.exists(output_img):
        os.mkdir(output_img)
    if not os.path.exists(output_npy):
        os.mkdir(output_npy)
    if not os.path.exists(output_eval):
        os.mkdir(output_eval)

    test_list_fp = open('%s/test.txt'%(output_npy),'w')
    val_list_fp = open('%s/val.txt'%(output_npy),'w')

    # if test_loader == None:
    test_list = './data/test_brats.txt'
    val_list = './data/val_brats.txt'
    test_set = BrainDataset(test_list,val_list,has_mean=True)
    n_test = len(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    baseline_pth = 'checkpoints/2020-08-28_16-23-33_best.pth'
    if reconstucter==None and segmenter==None:
        reconstucter = Reconstucter(n_channels=1, n_classes=1, bilinear=True)
        segmenter = Segmenter(n_channels=1, n_classes=2, bilinear=True)
        reconstucter.to(device=device)
        segmenter.to(device=device)
        logger.info("Loading model {}".format(model_pth))
        model_dict = torch.load(model_pth, map_location=device)
        reconstucter.load_state_dict(model_dict['reconstucter'])
        segmenter.load_state_dict(model_dict['segmenter'])
        logger.info("Model loaded !")
        
    segmenter_base = Segmenter(n_channels=1, n_classes=2, bilinear=True)
    segmenter_base.to(device=device)
    segmenter_base.load_state_dict((torch.load(baseline_pth, map_location=device))['segmenter'])
    
    reconstucter.eval()
    segmenter_base.eval()
    segmenter.eval()
    #找到全局diff的最大最小值
    diff_min = 0
    diff_max = 0
    ms_ssim = MS_SSIM(data_range=1.0,channel=1).to(device=device)
    softmax = nn.Softmax(dim=1)
    with tqdm(total=n_test//args.batch_size, desc='Stastic Round', unit='batch',ncols=120,ascii=True) as pbar:
        for i,(img, seg, seg_tumor, [name,slice_idx], [mean,std]) in enumerate(test_loader):
            img = img.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            mean = mean.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            std = std.unsqueeze(dim=1).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                img_no_tumor = reconstucter(img) #shape(batch,1,512,512)
                # seg_pred = segmenter(img_no_tumor)
            img = img * std + mean
            img_no_tumor = img_no_tumor * std + mean

            diff = img - img_no_tumor #diffrence between img and img_no_tumor
            diff = diff.cpu().numpy().astype(np.int32)
            tmp_min = diff.min(-1).min(-1).min(-1).min(-1)
            tmp_max = diff.max(-1).max(-1).max(-1).max(-1)  
            if tmp_max > diff_max:
                diff_max = tmp_max
            if tmp_min < diff_min:
                diff_min = tmp_min
            if i == 10:
                break 
            pbar.update()
    print('min:{}, max:{}'.format(diff_min,diff_max))
    diff_max = np.ones_like(diff) * diff_max
    diff_min = np.ones_like(diff) * diff_min
    
    #已找到全局最大最小值
    case_psnr = {}     
    with tqdm(total=n_test//args.batch_size, desc='Test Round', unit='batch',ncols=120,ascii=True) as pbar:
        for i,(img, seg, seg_tumor, [name,slice_idx], [mean,std]) in enumerate(test_loader):
            batch_size, h, w = img.shape[0], img.shape[-2], img.shape[-1]
            img = img.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            mean = mean.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            std = std.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            brain_mask = torch.where(std!=0,torch.ones_like(std),torch.zeros_like(std))
            
            seg_tumor_neg = torch.where(seg_tumor==1,torch.zeros_like(seg_tumor),torch.ones_like(seg_tumor))
            seg_tumor_neg = seg_tumor_neg.to(device=device, dtype=torch.long)
            seg_tumor = seg_tumor.unsqueeze(dim=1).numpy().astype(np.uint8)
            seg_for_save = seg.numpy().astype(np.uint8)
            with torch.no_grad():
                img_no_tumor = reconstucter(img) #shape(batch,1,512,512)
                seg_pred = segmenter(img_no_tumor)
            img_for_save = (img_no_tumor*brain_mask).squeeze().cpu().numpy().astype(np.float32)
            
            img = img * std + mean
            img_no_tumor = img_no_tumor * std + mean
            
            
            ####### 计算psnr #######
            mse = ((img-img_no_tumor)**2 * seg_tumor_neg).mean(dim=-1).mean(dim=-1).mean(dim=-1)
            psnr = 10*torch.log10(250*250/mse).cpu().numpy().astype(np.float32)
            ####### 计算psnr #######
            # _, _, specificity, sensitivity, _ = Evalution_Vessel(seg_pred,)
            # seg_pred = torch.sigmoid(seg_pred)
            # seg_pred = torch.where(seg_pred>0.5,torch.ones_like(seg_pred)*255,torch.zeros_like(seg_pred))
            seg_pred = softmax(seg_pred).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            # seg_for_save = seg_tumor.numpy().astype(np.uint8)

            diff = img - img_no_tumor #diffrence between img and img_no_tumor
            diff = diff.cpu().numpy().astype(np.int32)

            img = (img-img.min())/(img.max()-img.min())*brain_mask*255
            img_no_tumor = (img_no_tumor-img_no_tumor.min())/(img_no_tumor.max()-img_no_tumor.min())*brain_mask*255

            img = torch.cat((img,img_no_tumor),dim=3)
            img = img.cpu().numpy().astype(np.uint8)
            seg_tumor = np.concatenate((seg_tumor,np.zeros_like(seg_tumor)),axis=3)
            for b in range(batch_size):        
                #存储测试完的img与seg为npy格式   
                slice_pth = '%s/%s'%(output_npy,name[b])
                if not os.path.exists(slice_pth):
                    os.mkdir(slice_pth)
                np.save('%s/%s.npy'%(slice_pth,slice_idx[b]),{'brain':img_for_save[b],'seg_label':seg_for_save[b]})
                if name[b] in test_case_list:
                    test_list_fp.write('%s/%s.npy\n'%(slice_pth,slice_idx[b]))
                else:
                    val_list_fp.write('%s/%s.npy\n'%(slice_pth,slice_idx[b]))
                #存储结束
                #以case为单位创建文件夹
                case_path = '%s/%s'%(output_img,name[b])
                if not os.path.exists(case_path):
                    os.mkdir(case_path)
                if name[b] not in case_psnr:
                    case_psnr[name[b]] = [psnr[b]]
                else:
                    case_psnr[name[b]].append(psnr[b])
                
                mask_gt = np.zeros((3,h,2*w),dtype=np.uint8)
                mask_gt[1] = np.where(seg_tumor[b][0]==1,165,0) #G通道 
                mask_gt[2] = np.where(seg_tumor[b][0]==1,255,0) #B通道

                tmp = (cv2.cvtColor(img[b][0],cv2.COLOR_GRAY2RGB)).transpose((2,0,1))
                tmp_gt = 0.05*mask_gt + tmp

                tmp_diff = ((diff[b] + abs(diff_min[b]))/(diff_max[b]+abs(diff_min[b]))*255).astype(np.uint8)
                tmp_diff = (cv2.applyColorMap(tmp_diff[0], cv2.	COLORMAP_JET)).transpose(2,0,1)
               
                mask_rgb = (cv2.cvtColor(seg_pred[b],cv2.COLOR_GRAY2RGB)).transpose((2,0,1))
                mask_and_diff = np.concatenate((tmp_diff,mask_rgb),axis=2)
                total = np.concatenate((tmp_gt,mask_and_diff),axis=1)
                cv2.imwrite('{}/{}.jpg'.format(case_path,slice_idx[b]),total.transpose((1,2,0)))
                cv2.imwrite('{}/{}/{}_diff.jpg'.format(output,name[b],slice_idx[b]),tmp_diff[0])
            # if i == 10:
            #     break
            pbar.update()

    test_set = BrainDataset('data/test_brats.txt',has_mean=True)
    n_test = len(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    case_ssim = {}
    seg_base_tumor_sum = 0
    seg_recs_tumor_sum = 0   
    with tqdm(total=n_test//args.batch_size, desc='Eval Round', unit='batch',ncols=120,ascii=True) as pbar:
        for i,(img, seg, seg_tumor, [name,slice_idx], [mean,std]) in enumerate(test_loader):

            batch_size, h, w = img.shape[0], img.shape[-2], img.shape[-1]
            img = img.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            mean = mean.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            std = std.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
            seg_tumor_neg = torch.where(seg_tumor==1,torch.zeros_like(seg_tumor),torch.ones_like(seg_tumor))
            seg_tumor_neg = seg_tumor_neg.to(device=device, dtype=torch.int32)

            with torch.no_grad():
                img_no_tumor = reconstucter(img) #shape(batch,1,512,512)
                seg_base = segmenter_base(img)
                seg_recs = segmenter_base(img_no_tumor)

            img = img * std + mean
            img_no_tumor = img_no_tumor * std + mean
            
            ####### 计算healthiness #######
            ones = torch.ones((batch_size,h,w)).to(device=device, dtype=torch.float32)
            zeros = torch.zeros((batch_size,h,w)).to(device=device, dtype=torch.float32)
            seg_base = softmax(seg_base).argmax(dim=1)
            seg_recs = softmax(seg_recs).argmax(dim=1)
            # seg_base_tumor = torch.where(seg_base>0.5,ones,zeros)
            # seg_recs_tumor = torch.where(seg_recs>0.5,ones,zeros)
            # seg_base = torch.sigmoid(seg_base)
            # seg_recs = torch.sigmoid(seg_recs)
            # seg_base_tumor = torch.where(seg_base>0.5,ones,zeros)
            # seg_recs_tumor = torch.where(seg_recs>0.5,ones,zeros)
            seg_base_tumor_sum += seg_base.sum()
            seg_recs_tumor_sum += seg_recs.sum()
            ####### 计算healthiness #######

            seg_base_for_show = seg_base* 255
            seg_recs_for_show = seg_recs* 255

            img_cat = torch.cat((img,img_no_tumor),dim=3).squeeze()
            img_cat = img_cat.cpu().numpy().astype(np.int32)
            img_min = img_cat.min(-1).min(-1)
            img_max = img_cat.max(-1).max(-1)
            seg_cat = torch.cat((seg_base_for_show,seg_recs_for_show),dim=-1)
            seg_cat = seg_cat.squeeze().cpu().numpy().astype(np.uint8)

            for b in range(batch_size):   
                pred = ((img_no_tumor[b]*seg_tumor_neg[b])/1470).unsqueeze(dim=0) #这里要修改，主要是最小值最大值变化了
                target = ((img[b]*seg_tumor_neg[b])/1470).unsqueeze(dim=0)
                slice_ssim = (ms_ssim(pred,target)).cpu().numpy().astype(np.float32)
                if name[b] not in case_ssim:
                    case_ssim[name[b]] = [slice_ssim[0]]
                else:
                    case_ssim[name[b]].append(slice_ssim[0])
                img_cat[b] = ((img_cat[b] + abs(img_min[b]))/(img_max[b]+abs(img_min[b]))*255)
                img_for_save = np.concatenate((img_cat[b],seg_cat[b]),axis=0).astype(np.uint8)
                # img_for_save[] = img_for_save.cpu().numpy()
                #以case为单位创建文件夹
                case_path = '%s/%s'%(output_eval,name[b])
                if not os.path.exists(case_path):
                    os.mkdir(case_path)
                cv2.imwrite('{}/{}.jpg'.format(case_path,slice_idx[b]),img_for_save)

                
            # if i == 100:
            #     break
            pbar.update()

    total = 0
    tot_ssim = 0
    tot_heal = 0
    keys = list(case_ssim.keys())
    keys.sort()
    n_case = len(keys)
    for key in keys:
        case_psnr[key] = np.mean(case_psnr[key])
        case_ssim[key] = np.mean(case_ssim[key])
        total += case_psnr[key]
        tot_ssim += case_ssim[key]
    
    test_score = total/n_case
    healthiness = 1 - seg_recs_tumor_sum.type(torch.float) / seg_base_tumor_sum.type(torch.float)
    logger.info('Avr PSNR: {:.4f}, SSIM: {:.4f}, Healthiness: {:.4f}'.format(test_score,tot_ssim/n_case,healthiness))          
    # tb = pt.PrettyTable(['Case','PSNR','SSIM','Case','PSNR','SSIM'])
    # for key in keys:
    #     tb.add_row([key[4:],round(case_psnr[key],4),round(case_ssim[key],4),round(case_heal[key],4)])
    # print(tb)
    for m in range(n_case//4 +1 ): #每行打印四个
        for n in range(4):
            try:
                key = keys[m*4+n]
                print('{:<20s}: {:.4f}, {:.4f}; '.format(key,case_psnr[key],case_ssim[key]),end='')
            except:
                continue
        print('')
    test_list_fp.close()
    val_list_fp.close()

if __name__ == "__main__":
    args = get_args()
    logger = init_logging(starttime=time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime()))
    starttime='2020-08-29_10-47-27'
    # starttime='2020-09-07_19-19-10'
    # try:
    predict(args,logger=logger,starttime=starttime)
    # except KeyboardInterrupt:
    #     # pass
    #     try:
    #         os.remove(f'./log/{starttime}.txt')
    #     except:
    #         pass