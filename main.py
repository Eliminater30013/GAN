from input import *
from Optical_Properties import *


"""
Structure of the GANPOP:
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
In Essence:

        Blender->dataset->input.py->GANPOP[Train->Test]->Optical_Properties.py

"""
if __name__ == "main":
    print('hello')

# input.py
# name = "test6"
# test_size = 0.3
# split_dataset(experiment_name=name, test_size=test_size)
# Optical_Properties
# a = plot_loss_log('checkpoints/more_epoch/loss_log.txt')
# # TODO: Figure out hoe to smoothen out the loss graphs!
# epochs = [int(a[i]['epoch']) for i in range(len(a))]
# G_L1 = [float(a[i]['G_L1']) for i in range(len(a))]
# G_GAN = [float(a[i]['G_GAN']) for i in range(len(a))]
# D_real = [float(a[i]['D_real']) for i in range(len(a))]
# D_fake = [float(a[i]['D_fake']) for i in range(len(a))]
# plt.plot(epochs, G_L1,'o',  label='G_L1')
# #plt.ylim(0,0.02) # Use this to see the loss in detail
# #plt.plot(epochs, G_GAN, label=G_GAN)
# #plt.plot(epochs, D_real, label=D_real)
# #plt.plot(epochs, D_fake, label=D_fake)
# plt.show()
# exit()
# # AC and DC works (only sct and if trained with AC DC seperately), this time with light projection
# # Perhaps train with other AC patterns and see if AC GANPOP can do!
# # AC has spatial freq of f004 (0.035mm-1), phase 120. But should work with all AC patterns if trained
# # DC is f0 (0mm-1) phase 0. Should work with all phases.
# # Conex problem with vignetting, aim is to fix this in the future
# # test('./results/more_epoch/test_latest/images/010_fake_B.png',
# #     './results/more_epoch/test_latest/images/010_real_B.png')
# num = 20
# get_abs_and_sct_NMAE(f'./results/more_epoch/test_latest/images/{num:03}_fake_B.png',
#                      f'./results/more_epoch/test_latest/images/{num:03}_real_B.png', True)
# # test('./results/plain_AC_4/test_latest/images/010_fake_B.png',
# #     './results/plain_AC_4_copy/test_latest/images/010_real_B.png')
# get_abs_and_sct_NMAE(f'./results/plain_AC_4/test_latest/images/{num:03}_fake_B.png',
#                      f'./results/plain_AC_4/test_latest/images/{num:03}_real_B.png', True)
# # 300 epoch looks to be the best
# # exit()
# experiment_name_1 = 'plain_AC_4'
# experiment_name_2 = 'more_epoch'
# multiplier = 8
# TEST = True
# NMAE_values = {experiment_name_1: {'test': {}, 'train': {}},
#                experiment_name_2: {'test': {}, 'train': {}},
#                }
# # for i in range(1, 26):
# #     #     NMAE_values['test1'].update(
# #     #         {4 * i: get_abs_and_sct_NMAE(f'./checkpoints/test1/web/images/epoch{multiplier * i:03}_fake_B.png',
# #     #                                      f'./checkpoints/test1/web/images/epoch{multiplier * i:03}_real_B.png', False)})
# #     NMAE_values[experiment_name_1]['train'].update(
# #         {multiplier * i: get_abs_and_sct_NMAE(f'./checkpoints/test2/web/images/epoch{multiplier * i:03}_fake_B.png',
# #                                               f'./checkpoints/test2/web/images/epoch{multiplier * i:03}_real_B.png',
# #                                               False)})
# # print(len(NMAE_values[experiment_name_1]))
#
# list_results_1 = []
# list_results_2 = []
# # Use of WildCards to search in strings
# for counter1, filename in enumerate(
#         glob.iglob(f'./results/{experiment_name_1}/test_latest/images' + '*/*_B.png', recursive=True)):
#     list_results_1.append(filename)
# for counter2, filename in enumerate(
#         glob.iglob(f'./results/{experiment_name_2}/test_latest/images' + '*/*_B.png', recursive=True)):
#     list_results_2.append(filename)
# counter1 = int((counter1 + 1) / 2)
# counter2 = int((counter2 + 1) / 2)
# for i in range(counter1):
#     NMAE_values[experiment_name_1]['test'].update(
#         {i: get_abs_and_sct_NMAE(list_results_1[i],
#                                  list_results_1[i + 1], False)})
# for i in range(counter2):
#     NMAE_values[experiment_name_2]['test'].update(
#         {i: get_abs_and_sct_NMAE(list_results_2[i],
#                                  list_results_2[i + 1], False)})
# # Data whilst training
# y1 = [NMAE_values[experiment_name_1]['test'][i]['Sct_NMAE'] for i in range(0, counter1)]
# x1 = NMAE_values[experiment_name_1]['test'].keys()
# y2 = [NMAE_values[experiment_name_2]['test'][i]['Sct_NMAE'] for i in range(0, counter2)]
# x2 = NMAE_values[experiment_name_2]['test'].keys()
# # y2 = [NMAE_values[experiment_name_2]['train'][multiplier * i]['Sct_NMAE'] for i in range(1, 26)]
# # x2 = NMAE_values['train2'].keys()
# # plt.subplot(211)
# plt.plot(x1, y1, label=experiment_name_1)
# # plt.subplot(212)
# plt.plot(x2, y2, label=experiment_name_2)
# plt.legend(loc="upper right")
# plt.yscale('log')
# plt.show()
