import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import regex as re


## ADD error checking in future
# Aim to get both images identical via scaling
# path to absorption and reduced scattering optical property maps (grayscale images)
def show_gtruth(abs_path, sct_path):
    absorb = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
    sct = cv2.imread(sct_path, cv2.IMREAD_GRAYSCALE)
    sct = (sct / 255) * 2.5  # convert back to real values
    absorb = (absorb / 255) * 0.25  # convert back to real values
    plt.suptitle('Scattering + Absorption spectrum (in mm^-1)')
    plt.subplot(121)
    plt.imshow(absorb)
    plt.title('abs'), plt.xticks([]), plt.yticks([]), plt.colorbar()
    plt.subplot(122)
    plt.imshow(sct)
    plt.title('sct'), plt.xticks([]), plt.yticks([]), plt.colorbar()
    plt.show()


def test(gan_result_path, ground_truth_path):
    ## NEED TO SCALE IMAGES NOW AS BLENDER FILES CAN NOW BE LOADED IN TO PYCHARM

    # Pick fake and real images to find absorption + reduced scattering
    fake = cv2.imread(gan_result_path)
    real = cv2.imread(ground_truth_path)
    fake_copy = fake.copy()
    real_copy = real.copy()
    # Get scattering and absorption spectrum thx to Green/Red separation
    _, green_f, red_f = cv2.split(fake_copy)
    _, green_r, red_r = cv2.split(real_copy)
    # Convert to raw values
    green_r = (green_r / 255) * 2.5
    green_f = (green_f / 255) * 2.5
    red_f = (red_f / 255) * 0.25
    red_r = (red_r / 255) * 0.25
    # Original
    original_fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
    original_real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
    #### PLOTTING ####
    plt.suptitle('Scattering + Absorption spectrum (in mm^-1)')
    plt.subplot(331)
    plt.imshow(original_fake)
    plt.title('FAKE'), plt.xticks([]), plt.yticks([])
    #
    plt.subplot(332)
    w = plt.imshow(green_f)
    plt.colorbar(w)
    plt.title('Reduced scattering $\mu_s$`'), plt.xticks([]), plt.yticks([])
    plt.clim(0, 2.5)
    #
    plt.subplot(333)
    x = plt.imshow(red_f)
    plt.colorbar(x)
    plt.title('Absorption $\mu_a$'), plt.xticks([]), plt.yticks([])
    plt.clim(0, 0.25)
    #
    plt.subplot(334)
    plt.imshow(original_real)
    plt.title('REAL'), plt.xticks([]), plt.yticks([])
    #
    plt.subplot(335)
    y = plt.imshow(green_r)
    plt.colorbar(y)
    plt.title('Reduced scattering $\mu_s$`'), plt.xticks([]), plt.yticks([])
    plt.clim(0, 2.5)
    #
    plt.subplot(336)
    z = plt.imshow(red_r)
    plt.colorbar(z)
    plt.title('Absorption $\mu_a$'), plt.xticks([]), plt.yticks([])
    plt.clim(0, 0.25)
    ### Difference ###
    combined_diff = diff_images(fake, real)
    # print(f"NMAE: {NMAE(fake, real)}")
    abs_diff = diff_images(red_f, red_r)
    # print(f"Absolute NMAE: {NMAE(red_f, red_r)}")
    sct_diff = diff_images(green_f, green_r)
    # print(f"Reduced Scattering NMAE: {NMAE(green_f, green_r)}")
    # print(green_f[5][5], green_r[5][5], sct_diff[5][5])
    #
    plt.subplot(337)
    plt.imshow(combined_diff)
    plt.title('Combined Diff'), plt.xticks([]), plt.yticks([])
    #
    plt.subplot(338)
    z = plt.imshow(sct_diff)
    plt.colorbar(z)
    plt.title('Sct diff'), plt.xticks([]), plt.yticks([])
    plt.clim(0, 100)
    #
    plt.subplot(339)
    z = plt.imshow(abs_diff)
    plt.colorbar(z)
    plt.title('Abs Diff'), plt.xticks([]), plt.yticks([])
    plt.show()


# IMG1 comparison image, IMG2 ground truth
def diff_images(fake, real, show_image=False):
    diff = abs(fake - real) * 100 / real  # percentage
    if show_image:
        z = plt.imshow(diff)
        plt.colorbar(z)
        plt.title('Difference'), plt.xticks([]), plt.yticks([])
        plt.show()
    return diff


# Calculate the normalized mean absolute error of the GAN 'fake' and ground truth
# Predicted + groundtruth are np.array
def NMAE(predicted, ground_truth):
    error = np.sum(abs(predicted - ground_truth)) / np.sum(ground_truth)
    return error


def get_abs_and_sct_NMAE(gan_result_path, ground_truth_path, show=True):
    # Pick fake and real images to find absorption + reduced scattering
    fake = cv2.imread(gan_result_path)
    real = cv2.imread(ground_truth_path)
    # Get scattering and absorption spectrum thx to Green/Red separation
    _, green_f, red_f = cv2.split(fake)
    _, green_r, red_r = cv2.split(real)
    overall_NMAE = NMAE(fake, real)
    abs_NMAE = NMAE(red_f, red_r)
    sct_NMAE = NMAE(green_f, green_r)
    if show:
        print(f"Overall NMAE: {overall_NMAE}")
        print(f"Absolute NMAE: {abs_NMAE}")
        print(f"Reduced Scattering NMAE: {sct_NMAE}")
    return {'Avg_NMAE': overall_NMAE, 'Abs_NMAE': abs_NMAE, 'Sct_NMAE': sct_NMAE}


def plot_loss_log(path_file):
    loss_log = {}
    with open(path_file) as f:
        next(f)
        for count, line in enumerate(f):
            # get rid of unwanted characters like brackets, commas amd colon
            characters_to_remove = ":(),"
            pattern = "[" + characters_to_remove + "]"
            new_string = re.sub(pattern, "", line).split()
            # Store in dict
            loss_log[count] = dict((zip(new_string[::2], new_string[1::2])))
    return loss_log


a = plot_loss_log('checkpoints/more_epoch/loss_log.txt')
# TODO: Figure out hoe to smoothen out the loss graphs!
epochs = [int(a[i]['epoch']) for i in range(len(a))]
G_L1 = [float(a[i]['G_L1']) for i in range(len(a))]
G_GAN = [float(a[i]['G_GAN']) for i in range(len(a))]
D_real = [float(a[i]['D_real']) for i in range(len(a))]
D_fake = [float(a[i]['D_fake']) for i in range(len(a))]
plt.plot(epochs, G_L1,'o',  label='G_L1')
#plt.ylim(0,0.02) # Use this to see the loss in detail
#plt.plot(epochs, G_GAN, label=G_GAN)
#plt.plot(epochs, D_real, label=D_real)
#plt.plot(epochs, D_fake, label=D_fake)
plt.show()
exit()
# AC and DC works (only sct and if trained with AC DC seperately), this time with light projection
# Perhaps train with other AC patterns and see if AC GANPOP can do!
# AC has spatial freq of f004 (0.035mm-1), phase 120. But should work with all AC patterns if trained
# DC is f0 (0mm-1) phase 0. Should work with all phases.
# Conex problem with vignetting, aim is to fix this in the future
# test('./results/more_epoch/test_latest/images/010_fake_B.png',
#     './results/more_epoch/test_latest/images/010_real_B.png')
num = 20
get_abs_and_sct_NMAE(f'./results/more_epoch/test_latest/images/{num:03}_fake_B.png',
                     f'./results/more_epoch/test_latest/images/{num:03}_real_B.png', True)
# test('./results/plain_AC_4/test_latest/images/010_fake_B.png',
#     './results/plain_AC_4_copy/test_latest/images/010_real_B.png')
get_abs_and_sct_NMAE(f'./results/plain_AC_4/test_latest/images/{num:03}_fake_B.png',
                     f'./results/plain_AC_4/test_latest/images/{num:03}_real_B.png', True)
# 300 epoch looks to be the best
# exit()
experiment_name_1 = 'plain_AC_4'
experiment_name_2 = 'more_epoch'
multiplier = 8
TEST = True
NMAE_values = {experiment_name_1: {'test': {}, 'train': {}},
               experiment_name_2: {'test': {}, 'train': {}},
               }
# for i in range(1, 26):
#     #     NMAE_values['test1'].update(
#     #         {4 * i: get_abs_and_sct_NMAE(f'./checkpoints/test1/web/images/epoch{multiplier * i:03}_fake_B.png',
#     #                                      f'./checkpoints/test1/web/images/epoch{multiplier * i:03}_real_B.png', False)})
#     NMAE_values[experiment_name_1]['train'].update(
#         {multiplier * i: get_abs_and_sct_NMAE(f'./checkpoints/test2/web/images/epoch{multiplier * i:03}_fake_B.png',
#                                               f'./checkpoints/test2/web/images/epoch{multiplier * i:03}_real_B.png',
#                                               False)})
# print(len(NMAE_values[experiment_name_1]))

list_results_1 = []
list_results_2 = []
# Use of WildCards to search in strings
for counter1, filename in enumerate(
        glob.iglob(f'./results/{experiment_name_1}/test_latest/images' + '*/*_B.png', recursive=True)):
    list_results_1.append(filename)
for counter2, filename in enumerate(
        glob.iglob(f'./results/{experiment_name_2}/test_latest/images' + '*/*_B.png', recursive=True)):
    list_results_2.append(filename)
counter1 = int((counter1 + 1) / 2)
counter2 = int((counter2 + 1) / 2)
for i in range(counter1):
    NMAE_values[experiment_name_1]['test'].update(
        {i: get_abs_and_sct_NMAE(list_results_1[i],
                                 list_results_1[i + 1], False)})
for i in range(counter2):
    NMAE_values[experiment_name_2]['test'].update(
        {i: get_abs_and_sct_NMAE(list_results_2[i],
                                 list_results_2[i + 1], False)})
# Data whilst training
y1 = [NMAE_values[experiment_name_1]['test'][i]['Sct_NMAE'] for i in range(0, counter1)]
x1 = NMAE_values[experiment_name_1]['test'].keys()
y2 = [NMAE_values[experiment_name_2]['test'][i]['Sct_NMAE'] for i in range(0, counter2)]
x2 = NMAE_values[experiment_name_2]['test'].keys()
# y2 = [NMAE_values[experiment_name_2]['train'][multiplier * i]['Sct_NMAE'] for i in range(1, 26)]
# x2 = NMAE_values['train2'].keys()
# plt.subplot(211)
plt.plot(x1, y1, label=experiment_name_1)
# plt.subplot(212)
plt.plot(x2, y2, label=experiment_name_2)
plt.legend(loc="upper right")
plt.yscale('log')
plt.show()
