# GANPOP
Code, dataset, and trained models for a Generative Adversarial Network (GAN) created during the Nottingham Summer Engineering Research Programme. The GAN is able to detect the change in optical properties (GANPOP) and by utilising Spatial Frequency Domain Imaging (SFDI), it can detect malformed tissue structures indicative of early cancer.  



<img src="https://github.com/Eliminater30013/GAN/blob/main/imgs/Fig_1.jpg" width="512"/> 

## Setup

### Prerequisites

- Linux (Tested on Ubuntu 16.04)
- NVIDIA GPU (Tested on Nvidia P100 using Google Cloud)
- CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)
- Pytorch>=0.4.0
- torchvision>=0.2.1
- dominate>=2.3.1
- visdom>=0.1.8.3
- scipy

### Dataset Organization

All image pairs must be 256x256 and paired together in 256x512 images. '.png' and '.jpg' files are acceptable. Data needs to be arranged in the following order:

```bash
GANPOP_Pytorch # Path to all the code
└── Datasets # Datasets folder
      └── XYZ_Dataset # Name of your dataset
            ├── test
            └── train
```
<img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Figure2.jpg" width="512"/>

### Training

To train a model:
```
python train.py --dataroot <datapath> --name <experiment_name>  --gpu_ids 0 --display_id 0 --lambda_L1 60 --niter 100 --niter_decay 100 --pool_size 64 --loadSize 256 --fineSize 256 --gan_mode lsgan --lr 0.0002 --which_model_netG fusion
```
- To view epoch-wise intermediate training results, `./checkpoints/<experiment_name>/web/index.html`
- `--lambda_L1` weight of L1 loss in the cost function
- `--niter` number of epochs with constant learning rate 
- `--niter_decay` number of epochs with linearly decaying learning rate
- `--pool_size` number of past results to sample from by the discriminator
- `--lr` learning rate
- `--gan_mode` type of GAN used, either lsgan or vanilla
- `--which_model_netG` generator type; fusion, unet_256, or resnet_9blocks

<img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Network.jpg" width="512"/> 

### Pre-trained Models

Example pre-trained models for each experiment can be downloaded [here](https://drive.google.com/open?id=1Qyh3k0MTiSJqTVIJnZ1KNFERv8NWPkR3). 
- "AC" and "DC" specify the type of input images, and "corr" stands for profilometry-corrected experiment. 
- These models are all trained on human esophagus samples 1-6, human hands and feet 1-6, and 6 phantoms. 
- Test patches are available under `dataset` folder, including human esophagus 7-8, hands and feet 7-8, 4 ex-vivo pigs, 1 live pig, and 12 phantoms. To validate the models, please save the downloaded subfolders with models under `checkpoints` and follow the directions in the next section ("Testing").

### Testing

To test the model:
```
python test.py --dataroot <datapath> --name <experiment_name> --gpu_ids 0 --display_id 0 --loadSize 256 --fineSize 256 --model pix2pix --which_model_netG fusion
```
- The test results will be saved to a html file here: `./results/<experiment_name>/test_latest/index.html`.

### Dataset

The full-image dataset can be downloaded [here](https://drive.google.com/drive/folders/1o_hIv5xmkO1_jD34Jo6JD0V1kXm5SdiM?usp=sharing). Folders are structured in the same way as pre-trained models (AC and DC, with "corr" being profilometry-corrected). Please refer to README.txt for more details.


## Acknowledgments
- The GANPOP structure was created by Mason Chen and his fellows. Their research paper can be found here: 
[Chen, Mason T., et al. "GANPOP: Generative Adversarial Network Prediction of Optical Properties from Single Snapshot Wide-field Images." IEEE Transactions on Medical Imaging (2019).](https://ieeexplore.ieee.org/document/8943974)
- The code was also inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [FusionNet_Pytorch](https://github.com/GunhoChoi/FusionNet_Pytorch) which were the skeleton of the GANPOP architecture.


## Structure of the GANPOP
      1. Generate a DATASET, containing an input image (in) and a ground truth (out) that is 256*256*3
        a. Pair these images together to a 512*256*3 then split the image dataset to training (train) and
           testing (test).
        b. All dataset related functions can be found in input.py. Make sure to run input.py first if you
           wanted to create your own test/train folders. Otherwise follow the structure shown in README
    2. Now if you want to TRAIN your model, run the bash script [TODO: Make bash script] or type this into
       The terminal:
           " python train.py --dataroot <datapath> --name <experiment_name>  --gpu_ids 0 --display_id 0 --lambda_L1 60 
             --niter 100 --niter_decay 100 --pool_size 64 --loadSize 256 --fineSize 256 
             --gan_mode lsgan --lr 0.0002 --which_model_netG fusion " [SINGLE LINE]
       Where the <datapath> is the root path to your test/train dataset e.g. './datasets/DATA' , 
       Where DATA is dataset folder containing test/train
       And <experiment_name> = Your choice of experiment name.N.B. This will contain the discriminator and generator pth
       files (latest_net_D.pth or latest_net_G.pth) after training. If you haven't trained but instead want to just test
       certain models, please rename you discriminator and generator .pth files to latest_net_X.pth where X is either D
       or G respectively. 
    3. Now if you want TEST your model use [TODO: Make bash script] or type this into The terminal:
       " python test.py --dataroot <datapath> --name <experiment_name> 
       --gpu_ids 0 --display_id 0 --loadSize 256 --fineSize 256 --model pix2pix --which_model_netG fusion ".
       
                    FOR EXTRA OPTIONS FOR TRAINING AND TESTING SEE THE OPTIONS FOLDER!!!
        E.G. Note that if Load size and and fine size is changed to X then you can provide a X*X*3 image instead
        
    4. Check your training in the ./checkpoints folder and testing in the ./results folder.
    5. After testing, run Optical_Properties.py to see the optical properties of the image. Extra helper functions can
       also be found in this file as well
### In Essence:
        ideal epoch = 300 AC > DC
        Blender->dataset->input.py->GANPOP[Train->Test]->Optical_Properties.py
