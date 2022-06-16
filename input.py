import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import glob  # Unix System
from sklearn.model_selection import train_test_split
import argparse
import os
import shutil


# Count the number of files in folder, give a num variable if you have pairs (2) or triplets (3) of images
def count_files(path, num=1):
    return int(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]) / num)


# Split the image in path into r,g, b and return
def split_image(path, show_image=False):
    image = cv2.imread(path) # get image
    blue, green, red = cv2.split(image)
    if show_image: # show images if required
        cv2.imshow('blue', blue)
        cv2.imshow('green', green)
        cv2.imshow('red', red)
        cv2.waitKey(0) # wait for user input
    return blue, green, red


# Combine a red,blue,green image into one image and write image to workspace if required
def combine_images(red_path, green_path, blue_path=None, show_image=False, write=None):
    red_channel = cv2.imread(red_path, cv2.IMREAD_GRAYSCALE)  # red
    green_channel = cv2.imread(green_path, cv2.IMREAD_GRAYSCALE)  # green
    if blue_path:  # if we have blue path
        blue_channel = cv2.imread(green_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Sets the blue channel to 0, with the same data type as the red and green channel
        blue_channel = np.zeros_like(red_channel)
    combined = cv2.merge((blue_channel, green_channel, red_channel))
    if show_image: # show image of required
        cv2.imshow('r', red_channel)
        cv2.imshow('g', green_channel)
        cv2.imshow('b', blue_channel)
        cv2.imshow('combined', combined)
        cv2.waitKey(0)
    if write:  # save image if required
        cv2.imwrite(write, combined)
    return combined


# A resize function for SSOP result
def resize_SSOP(root_dir):
    root_dir = "Fin/in"
    for filename in glob.iglob(root_dir + '**/*.png', recursive=True):
        print(filename)
        im = Image.open(filename)
        imResize = im.resize((256,256), Image.ANTIALIAS)
        imResize.save(filename , 'png', quality=90)


# remove the blue parts of an image, useful when comparing Chen + Our data
def remove_blue(path_folder, output):
    for i, filename in enumerate(glob.glob(f'{path_folder}/*')):
        combine_images(filename, filename, write=f"{output}/{i+1:04}.png")


# Add blue to an image
def add_blue_C(path_folder, output):
    for i, filename in enumerate(glob.glob(f'{path_folder}/*')):
        combine_images(filename, filename,blue_path=filename, write=f"{output}/{i+1:04}.png")


# Stitch images together in a dataset to make complaint with GANPOP
def stitch_images(left_image, right_image, output_folder, counter):
    # array of imagers as input/ Can the output list as
    images = [left_image, right_image]
    widths, heights = zip(*(i.size for i in images))
    # get the width and height of image
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    Path(output_folder).mkdir(parents=True, exist_ok=True)  # save way of checking if directory exists, if not create it
    new_im.save(f'{output_folder}/{counter:03}.png')


# Load images to folder and return the image list
def load_images_in_folder(path_folder, ext="png"):  # extension, jpg,png etc
    image_list = []
    for filename in glob.glob(f'{path_folder}/*.{ext}'):
        im = Image.open(filename)
        image_list.append(im)

    return image_list


# Alternate elements in array
def alt_element(a, from_first=False):
    if from_first:
        return a[::2]
    else:
        return a[1::2]


# Get rid of vignetting
def normalize_by_reference(image_path, reference_path):
    height, width = reference_path.shape[:2]
    # generating vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(width, 150)
    kernel_y = cv2.getGaussianKernel(height, 150)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)

    test = reference_path.copy()
    for i in range(3):
        test[:, :, i] = test[:, :, i] / mask

    hsv = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # scale pixel values up or down for channel 1(Lightness)
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * 1.3  # scale pixel values up or down for channel 1(Lightness)
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    test = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Original_bright', image_path)
    cv2.imshow('Original_dark', reference_path)
    cv2.imshow('Result', test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Split dataset into training and testing datasets, ensure that in/in2/in3 already exists
def split_dataset(experiment_name, test_size=0.3, seed=None, no_gnd=False, not_random=False, no_blue=False,add_blue=False):
    # Requires in and out files
    folder = "in"
    if no_blue:
        Path(f"datasets/{experiment_name}/in2").mkdir(parents=True, exist_ok=True) # make folder if not exists
        remove_blue(f"datasets/{experiment_name}/in", f"datasets/{experiment_name}/in2")
        folder = "in2"
    if add_blue:
        Path(f"datasets/{experiment_name}/in3").mkdir(parents=True, exist_ok=True) # make folder if not exists
        add_blue_C(f"datasets/{experiment_name}/in", f"datasets/{experiment_name}/in3")
        folder = "in3"
    input_ac = load_images_in_folder(f"datasets/{experiment_name}/{folder}")
    # If you don't have gnd_truths then make a folder for just testing containing white output files
    test_path = f"datasets/{experiment_name}/test"
    train_path = f"datasets/{experiment_name}/train"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if no_gnd:
        for counter in range(len(input_ac)):
            # Ensure you have the white png file in main directory
            stitch_images(input_ac[counter], Image.open('white.png'), test_path, counter + 1)
        print(f'Testing dataset of size {counter + 1} was created')
        return
    # Otherwise use ground truths
    output_ac = load_images_in_folder(f"datasets/{experiment_name}/out")
    # Split test/train datasets
    if not_random:
        x_train, x_test, y_train, y_test = train_test_split(input_ac, output_ac, test_size=test_size, shuffle=False)

    else:
        x_train, x_test, y_train, y_test = train_test_split(input_ac, output_ac, test_size=test_size, random_state=seed)
    # Training samples
    for counter1, (left, right) in enumerate(zip(x_train, y_train)):
        stitch_images(left, right, train_path, counter1 + 1)
    # Testing samples
    for counter2, (left, right) in enumerate(zip(x_test, y_test)):
        stitch_images(left, right, test_path, counter2 + 1)

    print(f'Training dataset of size {counter1 + 1} and Testing dataset of size {counter2 + 1} were created')


# Rename files in a folder, useful for cross validation
def rename_files_CV(options):
    test_off = int(options.rename_with_offsets[0])
    train_off = int(options.rename_with_offsets[1])
    max_train = count_files(f"datasets/{options.name}/train")
    print(max_train)
    max_test = count_files(f"datasets/{options.name}/test")
    print(max_test)
    for i in range(max_test):
        os.rename(f"datasets/{options.name}/test/{i+1:03}.png", f"datasets/{options.name}/test/{i+test_off+1:03}.png")
    for i in range(max_train):
        os.rename(f"datasets/{options.name}/train/{i+1:03}.png", f"datasets/{options.name}/train/{i+train_off+1:03}.png")


# Main
if __name__ == "__main__":
    # Set up a Parser for Data Preparation
    parser = argparse.ArgumentParser(description='Run data preprocessing')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--test_size', type=float, default=0.3, help='Size of test dataset')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for shuffling train/test datasets. If '
                                                               'left default a new dataset will always be generated')
    parser.add_argument('--no_gnd', action='store_true', help='Enable this if you do not have a ground truth. Creates '
                                                              'test folder only')
    parser.add_argument('--not_random', action='store_true', help='Enable this if not random')
    parser.add_argument('--no_blue', action='store_true', help='Enable this if no blue channel')
    parser.add_argument('--add_blue', action='store_true', help='Enable this if want blue channel')
    parser.add_argument('--rename_with_offsets', '--list', nargs='+', help='Rename the files in folder with an offset'
                                                                           'First is test offset and second is train'
                                                                           'offset')
    options = parser.parse_args()
    split_dataset(experiment_name=options.dataset, test_size=options.test_size, seed=options.seed, no_gnd=options.no_gnd,
                  not_random=options.not_random, no_blue=options.no_blue, add_blue=options.add_blue)
    # Rename files if required
    if options.rename_with_offsets:
        rename_files_CV(options)
        exit()