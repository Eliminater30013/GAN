import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
from sklearn.metrics import mean_squared_error


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


# AC and DC works (only sct and if trained with AC DC seperately), this time with light projection
# Perhaps train with other AC patterns and see if AC GANPOP can do!
# AC has spatial freq of f004 (0.035mm-1), phase 120. But should work with all AC patterns if trained
# DC is f0 (0mm-1) phase 0. Should work with all phases.
# Conex problem with vignetting, aim is to fix this in the future
# test('./checkpoints/test1/web/images/epoch188_fake_B.png',
#      './checkpoints/test1/web/images/epoch188_real_B.png')
test('./results/test2/test_latest/images/015_fake_B.png',
     './results/test2/test_latest/images/015_real_B.png')


exit()
experiment_name_1 = 'test1'
experiment_name_2 = 'test2'
multiplier = 4
TEST = False
NMAE_values = {experiment_name_1: {},
               experiment_name_2: {},
               'train1': {},
               'train2': {}
               }
# for i in range(1, 51):
#     NMAE_values['test1'].update(
#         {4 * i: get_abs_and_sct_NMAE(f'./checkpoints/test1/web/images/epoch{multiplier * i:03}_fake_B.png',
#                                      f'./checkpoints/test1/web/images/epoch{multiplier * i:03}_real_B.png', False)})
#     NMAE_values['test2'].update(
#         {4 * i: get_abs_and_sct_NMAE(f'./checkpoints/test2/web/images/epoch{multiplier * i:03}_fake_B.png',
#                                      f'./checkpoints/test2/web/images/epoch{multiplier * i:03}_real_B.png', False)})

list_results = []
# Use of WildCards to dearch in strings
for filename in glob.iglob('./results/AC_test/test_latest/images' + '*/*_B.png', recursive=True):
    list_results.append(filename)
for i in range(0, len(filename) - 1):
    NMAE_values['train1'].update(
        {i: get_abs_and_sct_NMAE(list_results[i],
                                 list_results[i+1], False)})
    # NMAE_values['train2'].update(
    #     {j: get_abs_and_sct_NMAE(f'./results/test2/test_latest/images/{i:03}_fake_B.png',
    #                              f'./results/test2/test_latest/images/{i:03}_real_B.png', False)})
# Data whilst training
print(NMAE_values['train1'])

if TEST:
    y1 = [NMAE_values['test1'][4 * i]['Sct_NMAE'] for i in range(1, 51)]
    x = NMAE_values['test1'].keys()
    y2 = [NMAE_values['test2'][4 * i]['Sct_NMAE'] for i in range(1, 51)]
# Data whilst testing
else:
    y1 = [NMAE_values['train1'][i]['Sct_NMAE'] for i in range(0, 56)]
    x = NMAE_values['train1'].keys()
    y2 = [NMAE_values['train1'][i]['Sct_NMAE'] for i in range(0,  56)]
plt.plot(x, y1, label=experiment_name_1)
plt.plot(x, y2, label=experiment_name_2)
plt.legend(loc="upper right")
plt.yscale('log')
plt.show()
