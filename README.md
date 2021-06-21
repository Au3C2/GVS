<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="./data/imgs/brain.png" alt="Project logo"></a>
</p>

<h1 align="center"><strong>G</strong>enerator <strong>V</strong>ersus <strong>S</strong>egmentor</h1>

<p align="center"> An implementation for <strong>Generator Versus Segmentor: Pseudo-healthy Synthesis</strong>
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

This is an implementation for <strong>Generator Versus Segmentor: Pseudo-healthy Synthesis</strong>. 
[[arXiv]](https://arxiv.org/abs/2009.05722)

<p align="center" > <strong>Abstract</strong></p>

<p align="justify">In this paper, we discuss the problems of style transfer and artifacts respectively. To address these problems, we consider the local differences between the lesions and normal tissue. To achieve this, we propose an adversarial process that alternatively trains a generator and a segmentor. The segmentor is trained to distinguish the synthetic lesions (i.e. the region in synthetic images corresponding to the lesions in the pathological ones) from the normal tissue, while the generator is trained to deceive the segmentor and preserve the normal tissue at the same time. Qualitative and quantitative experimental results on public datasets of BraTS and LiTS demonstrate that the proposed method outperforms state-oftheart methods by preserving the style and removing the artifacts.</p>


<div align="center"><img src="./data/imgs/result2.jpg" width="75%"></div>
<div align="center"><img src="./data/imgs/result3.jpg" width="75%"></div>
<!-- ![result](./data/imgs/result.jpg) -->

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
|  test.py            # evaluate a model by some index, and extract images
|  README.md          # this file
|  main.py            # our main code, tarin our model on BraTS
|
|-- chechpoints        
   | pretrain.pth     # pretrined model
|
|-- data 
   |--test_npy        # test examples
   |  test_brats.txt  # text examples list
   |  train_brats.txt # train examples list    
|
|-- Results
   |--eval            # the pseudo-healthy images of test examples          
|
|-- unet
   |  unet_model.py          # store basic model
   |  unet_parts.py          # basic part of model
   |  network.py             # baseci part of model
|  
|-- utils
    |  dataset.py            # dataloader
    |  dice_loss.py          # calculate dice between two seg labels
    |  init_logging.py       # initial a logger to write a log
    |  ms_ssim.py            # calculate ms-ssim between two images
    |  nii2npy_brats.py      # split .nii in to .npy to train 
    |- split_cases_brats.py  # split cases into train/val/test set
```

## 🎈 Usage <a name="usage"></a>


* Prepare your LiTS/BraTS Dataset by following step:
  
  * train model by running ```main.py```
  * evaluate model by running ```test.py```


## ✍️ Authors <a name = "authors"></a>

- [@dazhangyu123](https://github.com/dazhangyu123)
- [@Au3C2](https://github.com/Au3C2)
- [@utdawn](https://github.com/utdawn)

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Spetial thanks to [@milesial](https://github.com/milesial). Part of this work is based on [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

## Citation
We hope that you find this work useful. If you would like to acknowledge us, please, use the following citation:
```
@misc{yunlong2020generator,
    title={Generator Versus Segmentor: Pseudo-healthy Synthesis},
    author={Zhang Yunlong, Lin Xin, Sun Liyan, Zhuang Yihong},
    year={2020},
    eprint={2009.05722},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```