# GANPOP: Generative Adversarial Network Prediction of Optical Properties
Code, dataset, and training models for a Generative Adversarial Network (GAN) produced during the Nottingham Summer Engineering Research Programme. The GAN is able to detect the changes in optical properties and by utilising Spatial Frequency Domain Imaging (SFDI), it can detect malformed tissue structures in rectangular and cylindrical geometries for simulated tissue. Adapted from Mason Chen's GANPOP [model](https://github.com/masontchen/GANPOP_Pytorch). A poster summarising my research can be found [here](https://github.com/Eliminater30013/GAN/blob/main/research/Research%20Poster.pdf)

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
Input-Output Example: 

<img src="https://github.com/Eliminater30013/GAN/blob/main/imgs/Figure2.jpg" width="512">

_Source: MasontChen_

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


GANPOP Architecture:

<img src="https://github.com/Eliminater30013/GAN/blob/main/imgs/Network.jpg" width="512"/> 

_Source: MasontChen_

### Pre-trained Models

Example pre-trained models for each experiment can be downloaded [here](https://drive.google.com/drive/folders/1eO_pzfuu87Mon0gTXrhw_rgr-41CbjWJ?usp=sharing).

### Testing

To test the model:
```
python test.py --dataroot <datapath> --name <experiment_name> --gpu_ids 0 --display_id 0 --loadSize 256 --fineSize 256 --model pix2pix --which_model_netG fusion
```
Ensure you type this in as a single line. 

- where <experiment_name> is the name of experiment containing .pth files          
- The test results will be saved to a html file here: `./results/<experiment_name>/test_latest/index.html`.

### Dataset

The full-image dataset can be downloaded [here](https://drive.google.com/drive/folders/1k24o3EtQVc5KYrjWpOQH3enuVdcXHmRj?usp=sharing).

### Blender Model: How the data was generated

The rectangular dataset was generated using *Rec.blend*, the rectangular tumour dataset was generated with *Rec_tumours.blend* and the cylindrical dataset was generated using *Cyl.blend*. The optical properties of the image were changed by varying final fac (the proportion between scattering and absorption), abs fact (the absorption factor) and the sct (the scattering factor). Ground truths were generated by varying the red and green channels whilst keeping the blue channel of the image at 0. Note that abs was correlated to red and sct was correlated to green. Blender Models were created By Jane Crowley, a link to her paper can be found [here](https://doi.org/10.1117/12.2576779)

For both datasets:

- `final fact` was varied between 0.05 and 0.95
- `abs` and `sct` fact was kept at 1.0.
- `R` and `G` channels were varied between 0.05 and 0.95

**N.B.** It may be easier to 'continue_train'/ transfer the learning from one of the already existing models provided as the GAN may take a couple of tries to produce the required output

Example:

If final fact is increased from 0.05->0.95 then R is decreased 0.95->0.05 and G is increased from 0.05->0.95. This is due to the proportions of abs:sct would change from 0.95:0.05 to 0.05:0.95

## In Essence
- Generate a DATASET (with Blender for example), containing an input image (in) and a ground truth (out) that is 256x256x3. Pair these images together to make a 512x256x3 then split the image dataset to training (train) and testing (test). All dataset related functions can be found in input.py. Make sure to run *input.py* first if you wanted to create your own test/train folders. 
- Train the model on the dataset and at the end of training a .pth file will be generated for both the generator and discriminator (latest_net_G.pth or latest_net_D.pth). 
- Once trained, the results will be saved `./checkpoints/<experiment_name>/web/index.html`
- Alternatively, if you haven't trained but instead just want to Test certain models ensure you have the latest .pth files stored in `./checkpoints/<experiment_name>`, or alternatively invoke `--which_epoch` option with the epoch.pth file e.g. for 20_net_\[D/G].pth simple type `--which_epoch 20`
- Once tested, the results will be saved `./results/<experiment_name>/test_latest/index.html`.
- After testing, run *Optical_Properties.py* to see the optical properties of the image. Extra helper functions can also be found in this file as well. 

## Pipeline

<img src="https://github.com/Eliminater30013/GAN/blob/main/imgs/concept.png">
        
## Acknowledgments
- The GANPOP structure was created by Mason Chen and his fellows. Their research paper can be found here: 
[Chen, Mason T., et al. "GANPOP: Generative Adversarial Network Prediction of Optical Properties from Single Snapshot Wide-field Images." IEEE Transactions on Medical Imaging (2019).](https://ieeexplore.ieee.org/document/8943974)
- The code was also inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [FusionNet_Pytorch](https://github.com/GunhoChoi/FusionNet_Pytorch) which were the skeleton of the GANPOP architecture.
