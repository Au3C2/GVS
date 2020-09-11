import numpy as np
import skimage.io as io
import os
from glob import glob
import random
import SimpleITK as sitk
from skimage import morphology,measure
from tqdm import tqdm
import scipy.ndimage as ndimage

down_scale = 0.5  # 横断面降采样因子
expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本
slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm

def get_boundingbox(mask):
        # mask.shape = [image.shape[0], image.shape[1], classnum]        
        # 删掉小于10像素的目标
        mask_without_small = morphology.remove_small_objects(mask,min_size=10,connectivity=2)
        # mask_without_small = mask
        # 连通域标记
        label_image = measure.label(mask_without_small)
        #统计object个数
        object_num = len(measure.regionprops(label_image))
        boundingbox = list()
        for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
            boundingbox.append(region.bbox)
        return object_num, boundingbox
def find995(casepaths):

    #################################
    # 随机挑选10例来统计0.995像素范围 #
    ################################# 
    
    casepaths_10 = random.sample(casepaths,10)
    count = np.zeros((500),dtype=np.float64)

    for i in tqdm(range(10),ascii=True,desc='Stastic Stage',ncols=120,position=0):
        Image = sitk.ReadImage(glob(f'{casepaths[i]}/*t2*')[0], sitk.sitkInt16)
        ArrayImage = sitk.GetArrayFromImage(Image)
        ArrayImage = ArrayImage//10
        for c in range(155):
            for h in range(240):
                for w in range(240):
                    count[ArrayImage[c,h,w]] += 1
    count = count/ (10* 155* 240* 240)
    for i in range(1,len(count),1):
        count[i] = count[i-1] + count[i]
        if count[i]>0.995:
            print(i)
            break
    return i*10

def slice_ct():
    rootpath = '/home/lx/MICCAI_BraTS_2019_Data_Training/HGG'
    outpath = '/home/lx/BraTS19_Norm/npy'
    casepaths = glob(rootpath+'/*')
    n_case = len(casepaths)
      
    # pixle995 = find995(casepaths)
    pixle995 = 1470 #先前测过，这里就直接赋值了 

    for i in tqdm(range(n_case),ncols=120,ascii=True):
        casename = casepaths[i].split('/')[-1]
        # print(f'processing case {casepaths[i]}')

        Image = sitk.ReadImage(glob(f'{casepaths[i]}/*t2*')[0], sitk.sitkFloat32)
        ArrayImage = sitk.GetArrayFromImage(Image)

        Label = sitk.ReadImage(glob(f'{casepaths[i]}/*seg*')[0], sitk.sitkUInt8)
        ArrayLabel = sitk.GetArrayFromImage(Label)

        # # 将灰度值在阈值之外的截断
        ArrayImage[ArrayImage > pixle995] = pixle995
        # ArrayImage[ArrayImage < -200] = -200
        brain_mask = np.where(ArrayImage>0,1,0)
        brain_area = brain_mask.sum(axis=-1).sum(axis=-1)
        # ######脑部区域归一化########
        brain_img = ArrayImage[ArrayImage>0]
        mean, std = np.mean(brain_img), np.std(brain_img, ddof=1)
        ArrayImage = (ArrayImage - mean)/ std * brain_mask

        #合并肿瘤区域
        # label_for_tumor = np.where(ArrayLabel>0,1,0)
        
        for n in range(len(ArrayImage)):
            if brain_area[n] > 0:
                slice_dict = {'brain':ArrayImage[n],'seg_label':ArrayLabel[n]}
                                # 'mean':mean,'std':std}
                path = f'{outpath}/{casename}'
                if not os.path.exists(path):
                    os.mkdir(path)

                filename = '{}/{:0>3d}.npy'.format(path,n)
                np.save(filename, slice_dict) 

if __name__ == "__main__":
    slice_ct()