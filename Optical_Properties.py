import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import regex as re
import argparse
from PIL import Image
import os,os.path

# ADD error checking in future
# Aim to get both images identical via scaling
# path to absorption and reduced scattering optical property maps (grayscale images)


def count_files(path):
    return int(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]) / 3)


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


def test(gan_result_path, ground_truth_path, options):
    # NEED TO SCALE IMAGES NOW AS BLENDER FILES CAN NOW BE LOADED IN TO PYCHARM

    # Pick fake and real images to find absorption + reduced scattering
    fake = cv2.imread(gan_result_path)
    real = cv2.imread(ground_truth_path)
    fake_copy = fake.copy()
    real_copy = real.copy()
    # Get scattering and absorption spectrum thx to Green/Red separation
    _, green_f, red_f = cv2.split(fake_copy)
    _, green_r, red_r = cv2.split(real_copy)
    # Convert to raw values
    if not options.scale_off:
        green_r = (green_r / 255) * 2.5
        green_f = (green_f / 255) * 2.5
        red_f = (red_f / 255) * 0.25
        red_r = (red_r / 255) * 0.25
    else:
        green_r = (green_r / 255)
        green_f = (green_f / 255)
        red_f = (red_f / 255)
        red_r = (red_r / 255)
    # Original
    original_fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
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
    w = plt.imshow(green_f)
    plt.colorbar(w)
    plt.title('Reduced scattering $\mu_s$`')
    plt.xticks([]), plt.yticks([])
    if not options.scale_off:
        plt.clim(0, 2.5)
    #
    plt.subplot(333)
    x = plt.imshow(red_f)
    plt.colorbar(x)
    plt.title('Absorption $\mu_a$')
    plt.xticks([]), plt.yticks([])
    if not options.scale_off:
        plt.clim(0, 0.25)
    #
    plt.subplot(334)
    plt.imshow(original_real)
    plt.title('REAL'), plt.xticks([]), plt.yticks([])

    plt.subplot(335)
    y = plt.imshow(green_r)
    plt.colorbar(y)
    plt.title('Reduced scattering $\mu_s$`')
    plt.xticks([]), plt.yticks([])
    if not options.scale_off:
        plt.clim(0, 2.5)
    #
    plt.subplot(336)
    z = plt.imshow(red_r)
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
    plt.title('Sct diff'), plt.xticks([]), plt.yticks([])
    plt.clim(0, 100)
    #
    plt.subplot(339)
    z = plt.imshow(abs_diff)
    plt.colorbar(z)
    plt.title('Abs Diff'), plt.xticks([]), plt.yticks([])
    plt.clim(0, 100)
    #
    plt.show()


# IMG1 comparison image, IMG2 ground truth
def diff_images(fake, real, show_image=False):
    # We will divide by 0 at times so ignore the warning
    np.seterr(divide='ignore', invalid='ignore')
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
    # We will divide by 0 at times so ignore the warning
    np.seterr(divide='ignore', invalid='ignore')
    error = np.sum(abs(predicted - ground_truth)) / np.sum(ground_truth)  # np.size() for MAE
    return error


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


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

    abs_rmse = rmse(red_f, red_r)
    sct_rmse = rmse(green_f, green_r)

    if show:
        print(f"Absolute RMSE: {abs_rmse}")
        print(f"Reduced Scattering RMSE: {sct_rmse}")
    return {'Abs_RMSE': abs_rmse, 'Sct_RMSE': sct_rmse}


def get_abs_and_sct_NMAE(gan_result_path, ground_truth_path, show=True):
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


def plot_loss_epoch(options):
    # TODO: figure out how to make argparser for OP, include lossvsepoch nmae and op
    a = plot_loss_log(f'checkpoints/{options.name}/loss_log.txt')
    epochs = [int(a[i]['epoch']) for i in range(len(a))]
    if options.loss_epoch == 'G_L1':
        loss = [float(a[i]['G_L1']) for i in range(len(a))]
    elif options.loss_epoch == 'G_GAN':
        loss = [float(a[i]['G_GAN']) for i in range(len(a))]
    elif options.loss_epoch == 'D_real':
        loss = [float(a[i]['D_real']) for i in range(len(a))]
    elif options.loss_epoch == 'D_fake':
        loss = [float(a[i]['D_fake']) for i in range(len(a))]
    else:
        print('WRONG INPUT! RAISE WARNING')
        exit()
    plt.plot(epochs, loss, 'o', label=('train: ' + options.loss_epoch))
    plt.legend(loc="upper right")
    # plt.ylim(0,0.02) # Use this to see the loss in detail
    plt.show()


def plot_loss_NMAE(options):
    experiment_name = options.name
    NMAE_values = {experiment_name: {'test': {}, 'train': {}}}
    list_train = []
    for counter1, filename in enumerate(
            glob.iglob(f'./checkpoints/{experiment_name}/web/images' + '*/*_B.png', recursive=True)):
        list_train.append(filename)
    if not list_train:
        print('Invalid Name')
        exit()
    counter1 = int((counter1 + 1) / 2)
    num = []
    for count, s in enumerate(list_train):
        if count % 2 == 0:
            continue
        result = re.search('epoch(.*)_', s)
        num.append(int(result.group(1)[0:3]))
    for i in range(counter1):
        NMAE_values[experiment_name]['train'].update(
            {num[i]: get_abs_and_sct_NMAE(list_train[i],
                                          list_train[i + 1], False)})
    list_test = []
    # Use of WildCards to search in strings
    for counter2, filename in enumerate(
            glob.iglob(f'./results/{experiment_name}/test_latest/images' + '*/*_B.png', recursive=True)):
        list_test.append(filename)
    counter2 = int((counter2 + 1) / 2)
    for i in range(counter2):
        NMAE_values[experiment_name]['test'].update(
            {i: get_abs_and_sct_NMAE(list_test[i],
                                     list_test[i + 1], False)})
    if options.NMAE_epoch[0] == 'train':
        task = 'train'
        counter = counter1
    elif options.NMAE_epoch[0] == 'test':
        task = 'test'
        counter = counter2
        num.clear()
        num = [i for i in range(counter)]
    else:
        print('RAISE warning test/train')
        exit()
    if options.NMAE_epoch[1] == 'sct':
        NMAE_type = 'Sct_NMAE'
    elif options.NMAE_epoch[1] == 'abs':
        NMAE_type = 'Abs_NMAE'
    elif options.NMAE_epoch[1] == 'avg':
        NMAE_type = 'Avg_NMAE'
    else:
        print('RAISE warning avg/sct/abs only')
        exit()
    y = [NMAE_values[experiment_name][task][i][NMAE_type] for i in num]
    x = NMAE_values[experiment_name][task].keys()

    plt.plot(x, y, label=(task + ': ' + NMAE_type))
    plt.legend(loc="upper right")
    plt.show()


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


def reject_outliers(data, m=4):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def plot_scatter(options):
    abso = []
    sct = []
    num = count_files(f'./results/{options.name}/test_latest/images/')
    print(num)
    for i in range(num):  # changed line so can get graph
        all = get_abs_and_sct_NMAE(f'./results/{options.name}/test_latest/images/{i+1:03}_fake_B.png',
                                   f'./results/{options.name}/test_latest/images/{i+1:03}_real_B.png', False)
        # all = get_abs_and_sct_RMSE(f'./results/{options.name}/test_latest/images/{i:03}_fake_B.png',
        #                            f'./results/{options.name}/test_latest/images/{i:03}_real_B.png', False)

        abso.append(all['Abs_NMAE'])
        sct.append(all['Sct_NMAE'])
        # abso.append(all['Abs_RMSE'])
        # sct.append(all['Sct_RMSE'])
    abso = np.array(abso)*100 #NMAE as a %
    sct = np.array(sct)*100 #NMAE as a %
    #abso = reject_outliers(abso)
    #sct = reject_outliers(sct)
    print("Average abs " + str(np.average(abso)) + "%")
    print("Average sct " + str(np.average(sct)) + "%")
    plt.plot(abso, sct, 'o',label="Dataset: " + options.name);
    #plt.legend()
    plt.xlabel('Normalised Absorption, $\mu_a$ (%)')
    plt.ylabel('Normalised Reduced Scattering, $\mu_s$` (%)')
    plt.title("Plot of Normalised Mean Absolute Error of $\mu_a$ and $\mu_s$`")
    #if options.name == 'gnd3_cyl_tum':
        #plt.xlim(0, 3)
        #plt.ylim(0, 3)
    #else:
        #plt.xlim(0, 8)
        #plt.ylim(0,8)
    # plt.title('Plot of Normalized Mean Absolute Error for $\mu_a$ & $\mu_s$` '), plt.xticks([]), plt.yticks([])
    # plt.xlabel("Abs_NMAE")
    # plt.ylabel("Abs_NMAE")
    # For plotting without tumours first 197
    # Test the tumours for rec_tumouts + cyl_tomours
    # if options.name == 'cyl_tumours' or options.name == 'rec_tumours':
    #     plt.scatter(abso[197:287], sct[197:287], color='black')
    #     plt.scatter(abso[287:307], sct[287:307], color='red')
    #     plt.scatter(abso[307:len(abso)], sct[307:len(sct)], color='green')
    # else:
    #     #plt.scatter(abso,sct)
    #     plt.scatter(abso[0:66], sct[0:66], color='red')
    #     plt.scatter(abso[66:132], sct[66:132], color='orange')
    #     plt.scatter(abso[132:199], sct[132:199], color='green')
    plt.show()
    plt.plot(abso, color='red',label="$\mu_a$")
    plt.plot(sct, color='green',label="$\mu_s$`")
    plt.legend(loc= "upper left")
    plt.xlabel('Sample Number, N')
    plt.ylabel('NMAE, as a %')
    plt.title("Plot of NMAE (as a %) of $\mu_a$ and  $\mu_s$` against N ")
    plt.show()


def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
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


def plot_RGB(options):
    all = []
    all2 = []
    abso = []
    sct = []
    abso2 = []
    sct2 = []
    for i in range(199):
        image = get_image(f'./results/{options.name}/test_latest/images/{i+1:03}_fake_B.png')
        image2 = get_image(f'./results/{options.name}/test_latest/images/{i + 1:03}_real_B.png')
        all.append(image[0][0])
        abso.append(image[0][0][0])
        sct.append(image[0][0][1])
        all2.append(image2[0][0])
        abso2.append(image2[0][0][0])
        sct2.append(image2[0][0][1])
    #print(all)

    #print(abso)
    #print(sct)
    plt.subplot()
    # Test the tumours for rec_tumouts + cyl_tomours
    if options.name == 'cyl_tumours' or options.name == 'rec_tumours':
        plt.scatter(abso[197:287], sct[197:287], color='black')
        plt.scatter(abso[287:307], sct[287:307], color='red')
        plt.scatter(abso[307:len(abso)], sct[307:len(sct)], color='green')
    else:
        plot1 = plt.figure(1)
        plt.scatter(abso[0:66], sct[0:66], color='red')
        plt.scatter(abso[66:132], sct[66:132], color='orange')
        plt.scatter(abso[132:198], sct[132:198], color='green')
        #plt.show()
        #plot2 = plt.figure(2)
        plt.scatter(abso2[0:66], sct2[0:66], color='red')
        plt.scatter(abso2[66:132], sct2[66:132], color='orange')
        plt.scatter(abso2[132:198], sct2[132:198], color='green')
        plt.show()
    #plt.plot(abso)
    plt.plot(abso2, color='red')
    #plt.plot(sct)
    plt.plot(sct2, color='green')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the optical properties of image OR '
                                                 'Plot NMAE vs epoch [train/test] OR'
                                                 'Plot loss vs epoch [train]')
    parser.add_argument('--name', type=str, help='Name of the Experiment', required=True)
    parser.add_argument('--loss_epoch', type=str, help='types of losses [G_L1, G_GAN, D_real, D_fake]')
    parser.add_argument('--NMAE_epoch', type=str, nargs='+',
                        help='Normalized mean absolute error for'
                             '[train/test] for [sct,abs,both]')
    parser.add_argument('--op', type=int, help='The sample number, see index.html in /results for more detail')
    parser.add_argument('--scale_off', action='store_true', help='Gets Rid of all scaling for abs,sct and abs,'
                                                                 'sct difference')
    parser.add_argument('--scatter', action='store_true', help="Plot a Scatter Diagram of abs vs sct")
    parser.add_argument('--check_rgb', action='store_true', help="Checks the RGB of image")
    options = parser.parse_args()

    if options.loss_epoch:
        plot_loss_epoch(options)
    if options.NMAE_epoch:
        plot_loss_NMAE(options)
    if options.op:
        plot_op(options)
    if options.scatter:
        plot_scatter(options)
    if options.check_rgb:
        plot_RGB(options)
