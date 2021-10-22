## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

This is the anonymous code of GVS, which mainly includes training details, pretrained model and the synthetic images of one volume.



## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

First check your enviroment.

```
pytorch >= 1.3.1
python >= 3.6
opencv-python >= 4.3
```

### File Tree

```
│  test.py           # evaluate a model by some index, and extract images
│  README.md                 # this file
│  main.py            # our main code, tarin our model on BraTS
│
|--chechpoints        
   | pretrain.pth     # pretrined model
|
|--data 
   |--test_npy        # test examples
   |  test_brats.txt  # text examples list
   |  train_brats.txt # train examples list    
|
|--Results
   |--eval            # the pseudo-healthy images of test examples          
|
├─unet
   │  unet_model.py          # store basic model
   │  unet_parts.py          # basic part of model
   |  network.py             # baseci part of model
│  
└─utils
    │  dataset.py            # dataloader
    │  init_logging.py       # initial a logger to write a log
    │  ms_ssim.py            # calculate ms-ssim between two images
    │  nii2npy_brats.py      # split .nii in to .npy to train 
    │  nii2npy_lits.py       # split .nii in to .npy to train 
    │  split_cases_brats.py  # split cases into train/val/test set
    │  split_cases_lits.py   # split cases into train/val/test set

```

## 🎈 Usage <a name="usage"></a>

* Then you can train model by running ```main.py```
* You can evaluate model by running ```test.py```

