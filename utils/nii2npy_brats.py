import numpy as np
import skimage.io as io
import os
from glob import glob
import random
import SimpleITK as sitk
from skimage import morphology,measure
from tqdm import tqdm
import scipy.ndimage as ndimage

def find995(casepaths):
    '''
    random choice 10 cases to cal 99.5% pixel range
    '''
    
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
    pixle995 = 1470 # for brats it's 1470 

    for i in tqdm(range(n_case),ncols=120,ascii=True):
        casename = casepaths[i].split('/')[-1]
        # print(f'processing case {casepaths[i]}')

        Image = sitk.ReadImage(glob(f'{casepaths[i]}/*t2*')[0], sitk.sitkFloat32)
        ArrayImage = sitk.GetArrayFromImage(Image)

        Label = sitk.ReadImage(glob(f'{casepaths[i]}/*seg*')[0], sitk.sitkUInt8)
        ArrayLabel = sitk.GetArrayFromImage(Label)

        # clamp the pixel in [0,pixle995]
        ArrayImage[ArrayImage > pixle995] = pixle995
        # ArrayImage[ArrayImage < -200] = -200
        brain_mask = np.where(ArrayImage>0,1,0)
        brain_area = brain_mask.sum(axis=-1).sum(axis=-1)
        ###### standerize brain area ########
        brain_img = ArrayImage[ArrayImage>0]
        mean, std = np.mean(brain_img), np.std(brain_img, ddof=1)
        ArrayImage = (ArrayImage - mean)/ std * brain_mask
        
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