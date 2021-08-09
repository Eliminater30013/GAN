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


