# GANPOP: Generative Adversarial Network Prediction of Optical Properties
Code, dataset, and training models for a Generative Adversarial Network (GAN) produced during the Nottingham Summer Engineering Research Programme. The GAN is able to detect the changes in optical properties and by utilising Spatial Frequency Domain Imaging (SFDI), it can detect malformed tissue structures in rectangular and cylindrical geometries for simulated tissue. 



<img src="https://github.com/Eliminater30013/GAN/blob/main/imgs/Fig_1.jpg" width="512"/> 

## Setup

### Prerequisites

- Linux (Tested on Ubuntu 20.04)
- NVIDIA GPU (Tested on Nvidia Geforce GTX 980)
- CUDA CuDNN
- Pytorch>=0.4.0
- torchvision>=0.2.1
- dominate>=2.3.1
- visdom>=0.1.8.3
- scipy

### Dataset Organization

All image pairs must be 256x256 and paired together in 256x512 images. '.png' and '.jpg' files are acceptable. Data needs to be arranged in the following order:

```bash
GANPOP # Path to all the code
└── datasets # Datasets folder
      └── XYZ_Dataset # Name of your dataset
            ├── test
            └── train
```
<img src="https://github.com/Eliminater30013/GAN/blob/main/imgs/Figure2.jpg" width="512"/>

### Training

To train a model:
```
python train.py --dataroot <datapath> --name <experiment_name>  --gpu_ids 0 --display_id 0 --lambda_L1 60 --niter 100 --niter_decay 100 --pool_size 64 --loadSize 256 --fineSize 256 --gan_mode lsgan --lr 0.0002 --which_model_netG fusion
```
Ensure you type this in as a single line. 
      
- To view epoch-wise intermediate training results, `./checkpoints/<experiment_name>/web/index.html`
- `<datapath>` root path to your test/train dataset e.g './datasets/XYZ_Dataset', where XYZ_Dataset is the dataset name
- `<experiment_name>` name of the experiment. N.B. This will contain the discriminator and generator .pth files
- `--lambda_L1` weight of L1 loss in the cost function
- `--niter` number of epochs with constant learning rate 
- `--niter_decay` number of epochs with linearly decaying learning rate
- `--pool_size` number of past results to sample from by the discriminator
- `--lr` learning rate
- `--gan_mode` type of GAN used, either lsgan or vanilla
- `--which_model_netG` generator type; fusion, unet_256, or resnet_9blocks
- See the options folder for more options when training/testing 



<img src="https://github.com/Eliminater30013/GAN/blob/main/imgs/Network.jpg" width="512"/> 

### Pre-trained Models

Example pre-trained models for each experiment can be downloaded [here](Insert shared drive folder). 

### Testing

To test the model:
```
python test.py --dataroot <datapath> --name <experiment_name> --gpu_ids 0 --display_id 0 --loadSize 256 --fineSize 256 --model pix2pix --which_model_netG fusion
```
Ensure you type this in as a single line. 

- where <experiment_name> is the name of experiment containing .pth files          
- The test results will be saved to a html file here: `./results/<experiment_name>/test_latest/index.html`.

### Dataset

The full-image dataset can be downloaded [here]. Folders are structured in the same way as pre-trained models. 

### Blender Model: How the data was generated


## In Essence
- Generate a DATASET (with Blender for example), containing an input image (in) and a ground truth (out) that is 256x256x3. Pair these images together to make a 512x256x3 then split the image dataset to training (train) and testing (test). All dataset related functions can be found in input.py. Make sure to run *input.py* first if you wanted to create your own test/train folders. 
- Train the model on the dataset and at the end of training a .pth file will be generated for both the generator and discriminator (latest_net_G.pth or latest_net_D.pth). 
- Once trained, the results will be saved `./checkpoints/<experiment_name>/web/index.html`
- Alternatively, if you haven't trained but instead just want to Test certain models ensure you have the latest .pth files stored in `./checkpoints/<experiment_name>`, or alternatively invoke --which_epoch option with the epoch.pth file e.g. for 20_net_\[D/G].pth simple type --which_epoch 20.
- Once tested, the results will be saved `./results/<experiment_name>/test_latest/index.html`.
- After testing, run *Optical_Properties.py* to see the optical properties of the image. Extra helper functions can also be found in this file as well. 

        Blender->dataset->input.py->GANPOP\[Train->Test]->Optical_Properties.py
        
## Acknowledgments
- The GANPOP structure was created by Mason Chen and his fellows. Their research paper can be found here: 
[Chen, Mason T., et al. "GANPOP: Generative Adversarial Network Prediction of Optical Properties from Single Snapshot Wide-field Images." IEEE Transactions on Medical Imaging (2019).](https://ieeexplore.ieee.org/document/8943974)
- The code was also inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [FusionNet_Pytorch](https://github.com/GunhoChoi/FusionNet_Pytorch) which were the skeleton of the GANPOP architecture.
