import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import glob  # Unix System


def split_image(path, show_image=False):
    image = cv2.imread(path)
    blue, green, red = cv2.split(image)
    if show_image:
        cv2.imshow('blue', blue)
        cv2.imshow('green', green)
        cv2.imshow('red', red)
        cv2.waitKey(0)
    return blue, green, red


def combine_images(red_path, green_path, blue_path=None, show_image=False, write=False):
    red_channel = red_path  # cv2.imread(red_path, cv2.IMREAD_GRAYSCALE)  # green
    green_channel = green_path  # cv2.imread(green_path, cv2.IMREAD_GRAYSCALE)  # red
    if blue_path:
        blue_channel = cv2.imread(blue_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Sets the blue channel to 0, with the same data type as the red and green channel
        blue_channel = np.zeros_like(red_channel)
    combined = cv2.merge((blue_channel, green_channel, red_channel))
    if show_image:
        cv2.imshow('r', red_channel)
        cv2.imshow('g', green_channel)
        cv2.imshow('b', blue_channel)
        cv2.imshow('combined', combined)
        cv2.waitKey(0)
    if write:
        cv2.imwrite("datasets/OLD/DC/test/06.png", combined)
    return combined


def stitch_images(left_image, right_image, output_folder, counter):
    # array of imagers as input/ Can the oyutput list as
    images = [left_image, right_image]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    Path(output_folder).mkdir(parents=True, exist_ok=True)  # save way of checking if directory exists, if not create it
    new_im.save(f'{output_folder}/{counter:03}.png')


def load_images_in_folder(path_folder, ext="png"):  # extension, jpg,png etc
    image_list = []
    for filename in glob.glob(f'{path_folder}/*.{ext}'):
        im = Image.open(filename)
        image_list.append(im)

    return image_list


def altElement(a, from_first=False):
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
    hsv[:, :, 1] = hsv[:, :, 1] * 1.3  ## scale pixel values up or down for channel 1(Lightness)
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * 1.3  ## scale pixel values up or down for channel 1(Lightness)
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    test = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Original_bright', image_path)
    cv2.imshow('Original_dark', reference_path)
    cv2.imshow('Result', test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow('Image', image_path)
    # cv2.imshow('Reference Image', reference_path)
    # result = image_path / reference_path
    # cv2.imshow('result', result)
    # cv2.waitKey(0)


# Train with even samples ie from_first = False, 2nd, 4th element
# Test with odd ie from_first = True 1st 3rd element
input_AC = load_images_in_folder("datasets/Blender/test3/in")
output_AC = load_images_in_folder("datasets/Blender/test3/out")
# input_AC = input_AC[:50]
# output_AC = output_AC[:70]
#input_AC = altElement(input_AC)
#output_AC = altElement(output_AC)
# Convert RGB to GRAY
# output_folder = "./z"
# for i in range(70):
#     opencvImage = cv2.cvtColor(np.array(input_AC[i]), cv2.COLOR_RGB2GRAY)
#     combined = combine_images(opencvImage, opencvImage)
#     Path(output_folder).mkdir(parents=True, exist_ok=True)  # save way of checking if directory exists, if not create it
#     cv2.imwrite(f'{output_folder}/{i:03}.png', combined)
# exit()

output_path = "datasets/Blender/test3/test"
num = 1
for left, right in zip(input_AC, output_AC):
    stitch_images(left, right, output_path, num)
    num = num + 1

## So we be changuingg
##fsdazdazsdf\sdf\d