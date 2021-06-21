import argparse
import os
import time
from unet.networks import define_G

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet.unet_model import *
from utils.dataset import BrainDataset
from utils.init_logging import init_logging


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, default='./checkpoints/pretrain.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batch_size')
    parser.add_argument('--output', '-o', metavar='OUTPUT', type=str, default='./Results/',
                        help='Filenames of output images', dest='output')
    parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int, help='witch gpu to use')
    return parser.parse_args()


def predict(args, logger=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    model_pth = args.model
    output = args.output

    output_img = output + '/img'
    output_npy = output + '/npy'
    output_eval = output + '/eval'
    if not os.path.exists(output):
        os.mkdir(output)
    if not os.path.exists(output_img):
        os.mkdir(output_img)
    if not os.path.exists(output_npy):
        os.mkdir(output_npy)
    if not os.path.exists(output_eval):
        os.mkdir(output_eval)

    # load model
    reconstucter = define_G(input_nc=1, output_nc=1, ngf=64, netG='resnet_9blocks', norm='instance')
    reconstucter.to(device=device)
    logger.info("Loading model {}".format(model_pth))
    model_dict = torch.load(model_pth, map_location=device)
    reconstucter.load_state_dict(model_dict['reconstucter'])
    logger.info("Model loaded !")

    reconstucter.eval()

    test_set = BrainDataset('data/test_brats.txt')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             drop_last=False)
    n_test = len(test_loader)

    bar = tqdm(enumerate(test_loader), total=n_test, desc='TEST ROUND', unit='batch', ncols=120, ascii=True)
    for i, data in bar:
        [img, _, seg_tumor, brain_mask, case_name, slice_idx] = data[0:6]
        img = img.unsqueeze(dim=1).to(device=device, dtype=torch.float32)
        brain_mask = brain_mask.numpy().astype(np.uint8)
        batch_size, h, w = img.shape[0], img.shape[-2], img.shape[-1]

        with torch.no_grad():
            img_no_tumor = reconstucter(img)  # shape(batch,1,512,512)

        img_ori = img.squeeze().cpu().numpy()
        img_rec = img_no_tumor.squeeze().cpu().numpy()
        seg_tumor = seg_tumor.numpy().astype(np.uint8)
        for b in range(batch_size):
            img_1 = (((img_ori[b] - img_ori[b].min()) / (img_ori[b].max() - img_ori[b].min()) * 255) * brain_mask[
                b]).astype(np.uint8)
            img_2 = (((img_rec[b] - img_ori[b].min()) / (img_ori[b].max() - img_ori[b].min()) * 255) * brain_mask[b])
            img_2 = img_2 * (img_2 <= 255) + 255 * (img_2 > 255)
            img_2 *= (img_2 > 0)
            img_2 = img_2.astype(np.uint8)

            label = cv2.applyColorMap((seg_tumor[b] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2RGB)
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_GRAY2RGB)

            diff = abs(img_ori[b] - img_rec[b])
            diff = (((diff - diff.min()) / (diff.max() - diff.min()) * 255) * brain_mask[b]).astype(np.uint8)
            diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

            img_up = np.concatenate((img_1, img_2), axis=1)
            img_down = np.concatenate((label, diff), axis=1)
            img_whole = np.concatenate((img_up, img_down), axis=0)

            # 以case为单位创建文件夹
            case_path = '%s/%s' % (output_eval, case_name[b])
            if not os.path.exists(case_path):
                os.mkdir(case_path)
            cv2.imwrite('{}/{}.jpg'.format(case_path, slice_idx[b]), img_whole)


if __name__ == "__main__":
    args = get_args()
    t = time.localtime()
    starttime = time.strftime("%Y-%m-%d_%H-%M-%S", t)
    starttime_r = time.strftime("%Y/%m/%d %H:%M:%S", t)  # readable time
    logger = init_logging(starttime, log_file=False)

    try:
        predict(args, logger=logger)
    except KeyboardInterrupt:
        logger.error('User canceled, start on %s, Saved interrupt.' % (starttime_r))