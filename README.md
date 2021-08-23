# Structure of the GANPOP
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

"""
