# Required Libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import regex as re
import argparse
from PIL import Image
import os, os.path
from scipy.stats import linregress

# Offsets + scaling if required. Can be done in Blender as well. Leave as [1,0,1,0] if no scaling needed
ABS_MUL = 1
ABS_OFF = 0
SCT_MUL = 1
SCT_OFF = 0
# EQ: ABS_MUL * x + ABS_OFF
# EQ: SCT_MUL * y + SCT_OFF


# Count the number of files in folder, give a num variable if you have pairs (2) or triplets (3) of images
def count_files(path, num=1):
    return int(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]) / num)


# Show the ground truth on matplotlib ie display
def show_gtruth(abs_path, sct_path):
    absorb = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)  # get  abs
    sct = cv2.imread(sct_path, cv2.IMREAD_GRAYSCALE)  # get sct
    sct = (sct / 255) * 2.5  # convert back to real values
    absorb = (absorb / 255) * 0.25  # convert back to real values
    # Display on matplotlib GUI
    plt.suptitle('Scattering + Absorption spectrum (in mm^-1)')
    plt.subplot(121)
    plt.imshow(absorb)
    plt.title('abs'), plt.xticks([]), plt.yticks([]), plt.colorbar()
    plt.subplot(122)
    plt.imshow(sct)
    plt.title('sct'), plt.xticks([]), plt.yticks([]), plt.colorbar()
    plt.show()


# Get the GAN and ground truth results and compare them on matplotlib GUI
def test(gan_result_path, ground_truth_path, options):
    # Pick fake and real images to find absorption + reduced scattering
    fake = cv2.imread(gan_result_path)
    real = cv2.imread(ground_truth_path)
    # Get scattering and absorption spectrum thx to Green/Red separation
    blue_f, green_f, red_f = cv2.split(fake)
    _, green_r, red_r = cv2.split(real)
    # Convert to raw values and do an appropriate pre-scaling to GAN result
    green_f = cv2.add(cv2.multiply(green_f, SCT_MUL), SCT_OFF)
    red_f = cv2.add(cv2.multiply(red_f, ABS_MUL), ABS_OFF)
    merged_fake = cv2.merge([blue_f, green_f, red_f])
    if not options.scale_off: # if we need to scale the Ground truth file to Chen's range
        green_r = (green_r / 255) * 2.5
        green_f = (green_f / 255) * 2.5
        red_f = (red_f / 255) * 0.25
        red_r = (red_r / 255) * 0.25
    else: # otherwise get raw result
        green_r = (green_r / 255)
        green_f = (green_f / 255)
        red_f = (red_f / 255)
        red_r = (red_r / 255)
    # Get the REAL and FAKE data ready to display
    original_fake = cv2.cvtColor(merged_fake, cv2.COLOR_BGR2RGB)
    #original_fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
    original_real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
    # ### Difference ###
    combined_diff = diff_images(fake, real)
    abs_diff = diff_images(red_f, red_r)
    sct_diff = diff_images(green_f, green_r)
    # PLOTTING
    plt.suptitle('Scattering + Absorption spectrum (in mm^-1)')
    plt.subplot(331)
    plt.imshow(original_fake)
    plt.title('FAKE'), plt.xticks([]), plt.yticks([])
    #
    plt.subplot(332)
    w = plt.imshow(green_r)
    plt.colorbar(w)
    plt.title('Reduced scattering $\mu_s$`')
    plt.xticks([]), plt.yticks([])
    if not options.scale_off:
        plt.clim(0, 2.5)
    #
    plt.subplot(335)
    x = plt.imshow(red_r)
    plt.colorbar(x)
    plt.title('Absorption $\mu_a$')
    plt.xticks([]), plt.yticks([])
    if not options.scale_off:
        plt.clim(0, 0.25)
    #
    plt.subplot(334)
    plt.imshow(original_real)
    plt.title('REAL'), plt.xticks([]), plt.yticks([])

    plt.subplot(333)
    y = plt.imshow(green_f)
    plt.colorbar(y)
    plt.title('Reduced scattering $\mu_s$`')
    plt.xticks([]), plt.yticks([])
    if not options.scale_off:
        plt.clim(0, 2.5)
    #
    plt.subplot(336)
    z = plt.imshow(red_f)
    plt.colorbar(z)
    plt.title('Absorption $\mu_a$')
    plt.xticks([]), plt.yticks([])
    if not options.scale_off:
        plt.clim(0, .25)
    #
    plt.subplot(337)
    plt.imshow(combined_diff.astype('uint8'))
    plt.title('Combined Diff'), plt.xticks([]), plt.yticks([])
    #
    plt.subplot(338)
    z = plt.imshow(sct_diff)
    plt.colorbar(z)
    plt.title('Sct diff')
    plt.xticks([]), plt.yticks([])
    plt.clim(0, 100)
    #
    plt.subplot(339)
    z = plt.imshow(abs_diff)
    plt.colorbar(z)
    plt.title('Abs Diff')
    plt.xticks([]), plt.yticks([])
    plt.clim(0, 100)
    #
    plt.show() # show the result


# Function used to check if the scale we used is correct. Fits a linear regression line to FAKE and REAL.
# If gradient is near 1 and y intercept is near 0 then we have used good enough scaling factors
def test2(options):
    all_red_f = []
    all_red_r = []
    all_green_r = []
    all_green_f = []
    num = count_files(f'./results/{options.name}/test_latest/images/')
    print(num)
    for x in range(num):
        # Pick fake and real images to find absorption + reduced scattering
        gan_result_path = f'./results/{options.name}/test_latest/images/{x+1:03}_fake_B.png'
        ground_truth_path = f'./results/{options.name}/test_latest/images/{x+1:03}_real_B.png'
        fake = cv2.imread(gan_result_path)
        real = cv2.imread(ground_truth_path)
        fake_copy = fake.copy()
        real_copy = real.copy()
        # Get scattering and absorption spectrum thx to Green/Red separation
        _, green_f, red_f = cv2.split(fake_copy)
        _, green_r, red_r = cv2.split(real_copy)
        # Shape it into 1d
        red_f = np.array(red_f)
        red_f = red_f.flatten()
        all_red_f.extend(red_f)

        red_r = np.array(red_r)
        red_r = red_r.flatten()
        all_red_r.extend(red_r)

        green_f = np.array(green_f)
        green_f = green_f.flatten()
        all_green_f.extend(green_f)

        green_r = np.array(green_r)
        green_r = green_r.flatten()
        all_green_r.extend(green_r)
    # Plot Abs then Sct and give the scaling and offset values
    plt.scatter(all_red_f, all_red_r)
    plt.plot(np.unique(all_red_f), np.poly1d(np.polyfit(all_red_f, all_red_r, 1))(np.unique(all_red_f)),
             color='red')
    print(linregress(all_red_f, all_red_r))
    plt.show()
    plt.scatter(all_green_f, all_green_r)
    plt.plot(np.unique(all_green_f), np.poly1d(np.polyfit(all_green_f, all_green_r, 1))(np.unique(all_green_f)),
             color='green')
    print(linregress(all_green_f, all_green_r))
    plt.show()


# Plot the Optical Property maps of one of the test images and get the NMAE and RMSE as %
def plot_op(options):
    # IF OVER 1000 FILES CHANGE PADDING TO 04 INSTEAD OF 03
    # Use this in conjunction with index.html in results of the experiment to see the oP
    if options.op > 999:
        print('WARNING options.op must be between 0-999 if over 1000 required change padding to 4')
    test(f'./results/{options.name}/test_latest/images/{options.op:03}_fake_B.png',
          f'./results/{options.name}/test_latest/images/{options.op:03}_real_B.png', options)
    get_abs_and_sct_NMAE(f'./results/{options.name}/test_latest/images/{options.op:03}_fake_B.png',
                         f'./results/{options.name}/test_latest/images/{options.op:03}_real_B.png', True)
    get_abs_and_sct_RMSE(f'./results/{options.name}/test_latest/images/{options.op:03}_fake_B.png',
                         f'./results/{options.name}/test_latest/images/{options.op:03}_real_B.png', True)


# Calculate the absorption and scattering Normalised Mean Absolute Error + return NMAE array
def get_abs_and_sct_NMAE(gan_result_path, ground_truth_path, show=True):
    # Pick fake and real images to find absorption + reduced scattering
    fake = cv2.imread(gan_result_path)
    real = cv2.imread(ground_truth_path)
    # Get scattering and absorption spectrum thx to Green/Red separation
    blue_f, green_f, red_f = cv2.split(fake)
    _, green_r, red_r = cv2.split(real)
    # new scaling
    green_f = cv2.add(cv2.multiply(green_f, SCT_MUL), SCT_OFF)
    red_f = cv2.add(cv2.multiply(red_f, ABS_MUL), ABS_OFF)
    # Chen's required scaling
    green_r = (green_r / 255) * 2.5
    green_f = (green_f / 255) * 2.5
    red_f = (red_f / 255) * 0.25
    red_r = (red_r / 255) * 0.25
    # NMAE calculation
    overall_NMAE = NMAE(real, real)
    abs_NMAE = NMAE(red_f, red_r)
    sct_NMAE = NMAE(green_f, green_r)
    # Show if needed
    if show:  # display as % return as decimal
        print(f"Overall NMAE: {overall_NMAE * 100:.2f}%")
        print(f"Absolute NMAE: {abs_NMAE * 100:.2f}%")
        print(f"Reduced Scattering NMAE: {sct_NMAE * 100:.2f}%")
    return {'Avg_NMAE': overall_NMAE, 'Abs_NMAE': abs_NMAE, 'Sct_NMAE': sct_NMAE}  # return the decimal values


# Calculate the absorption and scattering Root Mean Square Error + return RMSE array
def get_abs_and_sct_RMSE(gan_result_path, ground_truth_path, show=True):
    # Pick fake and real images to find absorption + reduced scattering
    fake = cv2.imread(gan_result_path)
    real = cv2.imread(ground_truth_path)
    # Get scattering and absorption spectrum thx to Green/Red separation
    _, green_f, red_f = cv2.split(fake)
    _, green_r, red_r = cv2.split(real)
    # new scaling
    green_r = (green_r / 255) * 2.5
    green_f = (green_f / 255) * 2.5
    red_f = (red_f / 255) * 0.25
    red_r = (red_r / 255) * 0.25
    # Get the RMSE for abs and sct
    abs_rmse = RMSE(red_f, red_r)
    sct_rmse = RMSE(green_f, green_r)
    # Show if needed
    if show:
        print(f"Absolute RMSE: {abs_rmse * 100:.2f}%")
        print(f"Reduced Scattering RMSE: {sct_rmse * 100:.2f}%")
    return {'Abs_RMSE': abs_rmse, 'Sct_RMSE': sct_rmse}  # return the decimal values


# Calculate the difference between two images, Return difference array
def diff_images(fake, real, show_image=False):
    # We will divide by 0 at times so ignore the warning
    np.seterr(divide='ignore', invalid='ignore')
    diff = abs(fake - real) * 100 / real  # percentage
    diff = np.nan_to_num(diff, neginf=0, posinf=0,nan=0) # to get rid of divide by 0 error
    # Show if needed
    if show_image:
        z = plt.imshow(diff)
        plt.colorbar(z)
        plt.title('Difference'), plt.xticks([]), plt.yticks([])
        plt.show()

    return diff


# Calculate the normalized mean absolute error of the GAN 'fake' and ground truth
# Predicted + ground-truth are np.array
def NMAE(predicted, ground_truth):
    # We will divide by 0 at times so ignore the warning
    np.seterr(divide='ignore', invalid='ignore')
    error = np.sum(abs(predicted - ground_truth)) / np.sum(ground_truth)  # np.size() for MAE
    error = np.nan_to_num(error, neginf=0, posinf=0,nan=0)
    return error


# Calculate the root mean square error of the GAN 'fake' and ground truth
# Predicted + ground-truth are np.array
def RMSE(predictions, ground_truth):
    return np.sqrt(((predictions - ground_truth) ** 2).mean())


# Plot the loss log seen in training
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


# Plot loss vs epoch seen in training. Loss can be [G_L1, G_GAN, D_real,D_fake]
def plot_loss_epoch(options):
    a = plot_loss_log(f'checkpoints/{options.name}/loss_log.txt')
    epochs = [int(a[i]['epoch']) for i in range(len(a))]
    # Check the loss required
    if options.loss_epoch == 'G_L1':
        loss = [float(a[i]['G_L1']) for i in range(len(a))]
    elif options.loss_epoch == 'G_GAN':
        loss = [float(a[i]['G_GAN']) for i in range(len(a))]
    elif options.loss_epoch == 'D_real':
        loss = [float(a[i]['D_real']) for i in range(len(a))]
    elif options.loss_epoch == 'D_fake':
        loss = [float(a[i]['D_fake']) for i in range(len(a))]
    else:
        print('WRONG INPUT!')
        exit()
    # Get the necessaryu data from loss files
    with open(f'{options.name}.txt', 'w') as f:
        for epoch, i_list in zip(epochs,loss):
            f.write("%s," % epoch)
            f.write("%s\n" % i_list)
    plt.plot(epochs, loss, 'o', label=('train: ' + options.loss_epoch))
    plt.legend(loc="upper right")
    # plt.ylim(0,0.02) # Use this to see the loss in detail
    plt.show()  # show when needed


# Find Outliers and remove them from data. Any data outside m*std will be removed
def reject_outliers(data1, data2,data3,data4, m=2):  # 2 is a fair number
    hot_key1 = abs(data1 - np.mean(data1)) < m * np.std(data1)  # find anomalies in abs
    data1 = data1[hot_key1]  # remove anomalies in abs
    data2 = data2[hot_key1]  # remove anomalies in sct
    data3 = data3[hot_key1]  # remove anomalies in abs rgb
    data4 = data4[hot_key1]  # remove anomalies in abs rgb
    hot_key2 = abs(data2 - np.mean(data2)) < m * np.std(data2)  # find anomalies in sct
    data1 = data1[hot_key2]  # remove anomalies in abs
    data2 = data2[hot_key2]  # remove anomalies in sct
    data3 = data3[hot_key2]  # remove anomalies in abs rgb
    data4 = data4[hot_key2]  # remove anomalies in abs rgb
    return data1, data2,data3,data4 # return cleaned data


# Plot a scatter plot of the data, giving the NMAE errors and showing the RGB content of data. REQUIRES PATH
def plot_scatter(options):
    abso = []
    sct = []
    path = "E:/ahmed/Latest_GANPOP"
    num = count_files(f"{path}/results/{options.name}/test_latest/images", 3) # count files
    print(num)
    for i in range(num):  # Get NMAE
        all = get_abs_and_sct_NMAE(f'./results/{options.name}/test_latest/images/{i + 1:03}_fake_B.png',
                                   f'./results/{options.name}/test_latest/images/{i + 1:03}_real_B.png', False)

        abso.append(all['Abs_NMAE'])
        sct.append(all['Sct_NMAE'])
    abso = np.array(abso) * 100  # NMAE as a %
    abso_rgb = []
    sct = np.array(sct) * 100  # NMAE as a %
    sct_rgb = []
    # Find the average RGB in data
    for i in range(num):
        image = get_image(f'./results/{options.name}/test_latest/images/{i + 1:03}_fake_B.png')
        abso_rgb.append(getAverageRGBN(image)[0])
        sct_rgb.append(getAverageRGBN(image)[1])
    # Convert to numpy array
    abso_rgb = np.array(abso_rgb)
    sct_rgb = np.array(sct_rgb)
    # Remove Anomalies if required
    abso, sct,abso_rgb,sct_rgb = reject_outliers(abso, sct, abso_rgb, sct_rgb)
    # Print
    print("Average abs " + str(np.average(abso)) + "%")
    print("Average sct " + str(np.average(sct)) + "%")
    # Display
    plt.plot(abso, sct, 'o', label="Dataset: " + options.name)
    plt.xlabel('Normalised Absorption, $\mu_a$ (%)')
    plt.ylabel('Normalised Reduced Scattering, $\mu_s$` (%)')
    plt.title("Plot of Normalised Mean Absolute Error of $\mu_a$ and $\mu_s$`")
    plt.show()
    plt.plot(abso, color='red', label="$\mu_a$")
    plt.plot(sct, color='green', label="$\mu_s$`")
    plt.plot(sct + abso, color='blue', label="$\mu_s$`+ $\mu_a$")
    plt.legend(loc="upper left")
    plt.xlabel('Sample Number, N')
    plt.ylabel('NMAE, as a %')
    plt.title("Plot of NMAE (as a %) of $\mu_a$ and  $\mu_s$` against N ")
    plt.show()
    # Plot the RGB vs error
    plt.plot(abso_rgb, abso,'o', color='red', label="$\mu_a$")
    plt.title("Plot of NMAE (as a %) of $\mu_a$ against R Channel Value of Test Dataset ")
    plt.xlabel('The R channel Value of Sample ')
    plt.ylabel('Normalised Absorption, $\mu_a$ (%)')
    plt.show()
    plt.plot(sct_rgb, sct, 'o', color='green', label="$\mu_s$'")
    plt.title("Plot of NMAE (as a %) of $\mu_s$' against G Channel Value of Test Dataset ")
    plt.xlabel('The G channel Value of Samples ')
    plt.ylabel('Normalised Reduced Scattering, $\mu_s$` (%)')
    plt.show()


# Get a numpy array of an image so that one can access values[x][y].
def get_image(image_path):
    image = Image.open(image_path, "r")
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == "RGB":
        channels = 3
    elif image.mode == "L":
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    return pixel_values


# Given PIL Image, return average value of color as (r, g, b)
def getAverageRGBN(image):
    # get image as numpy array
    im = np.array(image)
    # get shape
    w, h, d = im.shape
    # change shape
    im.shape = (w * h, d)
    # get average
    return tuple(np.average(im, axis=0))


# Plot the RGB of the data and show scaled result
def plot_RGB(options):
    abso = []
    v_abso = []
    sct = []
    v_sct = []
    abso2 = []
    sct2 = []
    num = count_files(f'./results/{options.name}/test_latest/images/')
    print(num)
    for i in range(num):
        # RGB
        image = get_image(f'./results/{options.name}/test_latest/images/{i + 1:03}_fake_B.png')
        image2 = get_image(f'./results/{options.name}/test_latest/images/{i + 1:03}_real_B.png')
        abso.append(getAverageRGBN(image)[0])
        abso2.append(getAverageRGBN(image2)[0])
        sct.append(getAverageRGBN(image)[1])
        sct2.append(getAverageRGBN(image2)[1])
        # Abs and Sct Values
        val = get_abs_and_sct_NMAE(f'./results/{options.name}/test_latest/images/{i + 1:03}_fake_B.png',
                                   f'./results/{options.name}/test_latest/images/{i + 1:03}_real_B.png', False)

        v_abso.append(val['Abs_NMAE'])
        v_sct.append(val['Sct_NMAE'])
    v_abso = np.array(v_abso) * 100  # NMAE as a %
    v_sct = np.array(v_sct) * 100  # NMAE as a %
    print("Average abs " + str(np.average(v_abso)) + "%")
    print("Average sct " + str(np.average(v_sct)) + "%")
    # Display
    n_sct = cv2.add(cv2.multiply(np.array(sct), SCT_MUL), SCT_OFF)
    n_abso = cv2.add(cv2.multiply(np.array(abso), ABS_MUL), ABS_OFF)
    plt.plot(n_abso, label="Scaled Abs of Chen (real)")
    plt.plot(abso, label="Abs of Chen (real)")
    plt.plot(abso2, color='red', label="Abs of Ahmed(arbitrary)")
    plt.plot(n_sct, label="Scaled Sct of Chen(real)")
    plt.plot(sct, label="Sct of Chen(real)")
    plt.plot(sct2, color='green', label="Sct of Ahmed(arbitrary)")
    plt.legend()
    plt.show()
    # Scale
    abso2 = np.array(abso2) * (0.25/255)
    abso = np.array(abso) * (0.25 / 255)
    sct2 = np.array(sct2) * (2.5/255)
    sct = np.array(sct) * (2.5 / 255)
    # And save to file for easier comparison
    np.savetxt('abs.out', abso2, delimiter=',')
    np.savetxt('sct.out', sct2, delimiter=',')
    np.savetxt('abs_chen.out', abso, delimiter=',')
    np.savetxt('sct_chen.out', sct, delimiter=',')


# Main, Create an Argument Parser to control code via cmd line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the optical properties of image OR '
                                                 'Plot NMAE vs epoch [train/test] OR'
                                                 'Plot loss vs epoch [train]')
    parser.add_argument('--name', type=str, help='Name of the Experiment', required=True)
    parser.add_argument('--loss_epoch', type=str, help='types of losses [G_L1, G_GAN, D_real, D_fake]')

    parser.add_argument('--op', type=int, help='The sample number, see index.html in /results for more detail')
    parser.add_argument('--scale_off', action='store_true', help='Gets Rid of all scaling for abs,sct and abs,'
                                                                 'sct difference')
    parser.add_argument('--scatter', action='store_true', help="Plot a Scatter Diagram of abs vs sct")
    parser.add_argument('--check_rgb', action='store_true', help="Checks the RGB of image")
    options = parser.parse_args()
    # Options
    if options.loss_epoch:
        plot_loss_epoch(options)
    if options.op:
        plot_op(options)
    if options.scatter:
        plot_scatter(options)
    if options.check_rgb:
        plot_RGB(options)
