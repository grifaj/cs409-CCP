# from click import edit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import os
# from wand.api import library
# import wand.color
# import wand.image
import numpy as np
from skimage import measure
from PIL import Image
from scipy import ndimage
import random
import pandas as pd
from character_translation_load import DatasetLoad
from tqdm import tqdm
import scipy
from tqdm import tqdm
from character_translation_load import DatasetLoad


# CONSTANTS
DATA_DIR = '/dcs/20/u2002183/cs409-CCP-1/seal-script-images'
SAVE_DIR = '/dcs/large/u2009169/seal-script-images'
IMG_FILETYPE = '.png'
test = True

# The character index to edit up to
edit_index = 10

# The number of variants to obtain for each character
DESIRED_VARIANTS = 100


def load_image(file):
#     img_folder = file[0:file.index('_')]
    img = Image.open(file).convert('L')
    
    im = np.asarray(img)
        
    return im


def resize_all_images():
    for root, dirs, files in os.walk(DATA_DIR):
        for name in files:
            if '.png' in name:
#                 print(name[:name.index(".")])
                im = load_image(name[:name.index(".")])
                resized_image = pad_image((512,512), im.shape, im)
                plt.imsave(os.path.join(root, name), resized_image, cmap='gray')
                print(f'Saved resized image at {os.path.join(root, name)}')


def downsize_images(csvFile, i, new_height, new_width):
    data = pd.read_csv(csvFile, names = ["img", "label"])
    print(data)
    for x in tqdm(range(len(data["img"]))):
        if data["label"][x] > i:
            break
        im = Image.open(data["img"][x]).convert("L")
        img = im.resize((new_height, new_width))
        img.save(data["img"][x])
        
        
def binarise_image(im, thresh):
    idx = (im < thresh)
    
    binary_im = np.zeros(im.shape, dtype='int')
    
    binary_im[idx] = 255
    binary_im[~idx] = 0
    
    return binary_im


def extract_character(im):
    binary_im = binarise_image(im, 200) # Input images should be black & white anyway, so 128 is arbitrary

    # threshold at 200
    threshed = np.zeros(im.shape, 'int')
    threshed[im<200] = 1
    
    comps = measure.label(threshed, background=0)
    
    component_map, labels = comps, np.unique(comps)

    min_x = im.shape[0]
    min_y = im.shape[1]
    max_x = 0
    max_y = 0
    for label in labels[1:]:
        char_box = char_bounding_box(component_map, label)
        min_x = np.min(np.array([min_x, char_box[0]]))
        min_y = np.min(np.array([min_y, char_box[1]]))
        max_x = np.max(np.array([max_x, char_box[2]]))
        max_y = np.max(np.array([max_y, char_box[3]]))

    extracted_char = np.array(im[min_y:max_y, min_x:max_x])

    return extracted_char
    
    
# find the coordinate bounding box of a given label in a components image
def char_bounding_box(comps, label=1):
    
    # array of image coordinates in x and y
    xx, yy = np.meshgrid(np.arange(0,comps.shape[1]), np.arange(0,comps.shape[0]))

    # mask/select by where value is given label (component)
    where_x = xx[comps==label]
    where_y = yy[comps==label]
    
    # find min and max extents of coordinates
    return np.min(where_x), np.min(where_y), np.max(where_x), np.max(where_y)   


# Return image im with size shape_from after padding with 255 (white) to size shape_to
def pad_image(shape_to, shape_from, im):
#     print(f"Original size: {im.shape}")
    padded_image = np.pad(im, ((500,500),(500,500)), 'constant', constant_values=(255))

    if shape_from[0] > shape_to[0] and shape_from[1] > shape_to[1]:
        cropped_image = Image.fromarray(im).resize(shape_to)
        cropped_image = np.asarray(cropped_image)
    else:
        cropped_image = padded_image[padded_image.shape[0]//2-shape_to[0]//2:padded_image.shape[0]//2+shape_to[0]//2, padded_image.shape[1]//2-shape_to[1]//2:padded_image.shape[1]//2+shape_to[1]//2]
#     print(f"Cropped size: {cropped_image.shape}")
    return cropped_image


def rotate_char(im, std, rot):
    
    extracted_char = extract_character(im)
    rotated_char = ndimage.rotate(extracted_char, rot, reshape=True, mode='constant', cval=255)
    
#     if rotated_char.shape[0] < im.shape[0] and rotated_char.shape[1] < im.shape[1]:
    rotated_image = pad_image((im.shape[0],im.shape[1]), rotated_char.shape, rotated_char)
#     else:
#         rotated_image = rotated_char
    
    return rotated_image


def add_gaussian_noise(img, std, rot):
    noise = np.random.normal(0, std, img.shape) 

    # Add the noise to the image
    img_noised = img + noise

    # Clip the pixel values to be between 0 and 255.
    img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
    
    return img_noised


def add_gaussian_blur(img, std, rot):
    std = int(std//10)
    gauss_blurred = ndimage.gaussian_filter(img, std)
    
    return gauss_blurred


def remove_block(img, std, rot):
    block_size = 20
    extracted_char = extract_character(img)
#     print(extracted_char.shape)
    block_size = min(block_size, extracted_char.shape[0])
    block_size = min(block_size, extracted_char.shape[1])
    rand_x = random.randint(0, extracted_char.shape[0]-block_size)
    rand_y = random.randint(0, extracted_char.shape[1]-block_size)
#     print(rand_x, rand_y)
    extracted_char[rand_x:rand_x+block_size, rand_y:rand_y+block_size] = 255
    
    
    padded_image = pad_image((img.shape[0],img.shape[1]), extracted_char.shape, extracted_char)
    return padded_image


def phase_swap(img, std, rot):
    A = img
    
    # now dft and swap phases
    dft_A = np.fft.fft2(A)
    mag_A = np.abs(dft_A)
    angle_A = np.angle(dft_A)
    
    ref_img = Image.open(os.path.join(DATA_DIR, 'zebra.gif')).convert('L')
    ref_im = np.asarray(ref_img)
    ref_img_resized = ref_img.resize((img.shape[0], img.shape[1]))
    dft_B = np.fft.fft2(ref_img_resized)  # type:ignore
    mag_B = np.abs(dft_B)
    angle_B = np.angle(dft_B)

    dft_B2 = mag_B * np.exp(1j*angle_A)

    # inverse dft
    B2 = np.fft.ifft2(dft_B2).real
    
    return B2


def flip_image(img, std, rot):
    im_flipped = np.flip(img, axis=1)
    return im_flipped


def squish_image(img, _, stretch):
    """Stretches (or really squishes) an image in either x or y by a percentage"""
    HWEIGHT = 1.0
    VWEIGHT = 1.1
    inH = stretch > 0
    stretch = abs(stretch)+10 # always squish at least 10 percent and at most 40, before weights
    extracted_char = extract_character(img)
    h,w = extracted_char.shape[:2]
    if inH:
        factor = 100 - (10 + (stretch - 10)*HWEIGHT)
        h = int(np.ceil(h * (factor/100)))
    else:
        factor = 100 - (10 + (stretch - 10)*VWEIGHT)
        w = int(np.ceil(w * (factor/100)))
    stretched_char = np.array(Image.fromarray(extracted_char).resize((w,h)))
    padded_image = pad_image(img.shape[:2], stretched_char.shape, stretched_char)
    return padded_image
    


def add_effects(im):
    ### Function parameters:
    #   im - the input image
    #   std - the standard deviation of gaussian filters, 1 <= std <= 100
    #   rot - the degree of rotation, -90 <= rot <= 90

    effects = { # CAN ADD HORIZONTAL FLIP
        1: rotate_char,
        2: remove_block,
        #3: phase_swap,
        4: flip_image,
        5: add_gaussian_blur,
        6: add_gaussian_noise,
        7: squish_image
    }
    
    effect_transform = [
        rotate_char, remove_block, flip_image, squish_image
    ]
    
    effect_pixel = [
        add_gaussian_blur, add_gaussian_noise
    ]
        
    numTransformEffects = random.randint(0, len(effect_transform))
    numPixelEffects = random.randint(0, len(effect_pixel))
    if numTransformEffects + numPixelEffects == 0:
        numTransformEffects += 1
        
    # num_effects = random.randint(1, len(effects)-2)
    effect_image = np.copy(im)
    
    for _ in range(numTransformEffects):
        effect_image = random.choice(effect_transform)(effect_image, random.randrange(10, 50, 10), random.randint(-30, 30))
    for _ in range(numPixelEffects):
        effect_image = random.choice(effect_pixel)(effect_image, random.randrange(10, 50, 10), random.randint(-30, 30))
    
    return effect_image
        
    # for f in range(1, num_effects+1):
    # for _ in range(num_effects):
        # print(f"Effect: {f}")
        # f= random.randint(1, len(effects))
        # effect_image = effects[f](effect_image, random.randrange(10, 50, 10), random.randint(-30, 30))
        # 
    # return effect_image



def make_variant(im_filename, variant_filename):
    im = load_image(im_filename)
    
    # Enforce images are all 128x128
    padded_im = pad_image((128,128), im.shape, im)
    
    # Add distortions to the image
    edited_image = add_effects(padded_im)
    
    # Save image as grayscale
    plt.imsave(variant_filename, edited_image, cmap='gray')


def get_data_csv(edit_index):
    if not os.path.exists(os.path.join(DATA_DIR, 'trainData.csv')):
        DatasetLoad(DATA_DIR, edit_index, 'trainData.csv').createCsv() 
        
        
def get_data_csv_override(edit_index):
    if os.path.exists(os.path.join(DATA_DIR, 'trainData.csv')):
        os.remove(os.path.join(DATA_DIR, 'trainData.csv'))
    DatasetLoad(DATA_DIR, edit_index, 'trainData.csv').createCsv() 


def main(start_index=1):
    if os.path.exists(os.path.join(DATA_DIR, 'trainData.csv')):
        df = pd.read_csv(os.path.join(DATA_DIR,'trainData.csv'), sep=",", names = ["img", "label"])
        labels = np.unique(np.asarray(df["label"], dtype=int))
        name = df[df["label"] == 1]["img"][0][:]
        # Loop over characters
        print(labels)
        for i in labels[labels >= start_index]:
            print(i)
            variant_num = 1

            # Number of images currently in folder for character i
            # num_variants = len(df[df["label"] == i])
            directory = SAVE_DIR + '/' + str(i) + '/'
            num_variants = len([name for name in os.listdir(directory) if (os.path.isfile(os.path.join(directory, name)) and IMG_FILETYPE in name)])

            # Calculate number of variant images to make for each image in folder
            div = len(df[df["label"] == i])
            num = DESIRED_VARIANTS - num_variants
            if num <= 0:
                variants_to_make = [0 for _ in range(num_variants)]
            else:
                variants_to_make = ([num // div + (1 if x < num % div else 0)  for x in range (div)])

            # Loop over images in character folder
            for j in range(div):
                # Make required number of variants for each image
                for k in range(variants_to_make[j]):
                    # Make filename for new image
                    variant_filename = SAVE_DIR + '/' + str(i) + '/' + str(i) + '_' + str(num_variants+variant_num) + IMG_FILETYPE

                    # Make new variant image
                    make_variant(df[df["label"] == i].iloc[j]["img"], variant_filename)

                    # Increment to get next variant number for next new image filename
                    variant_num += 1
        print('[INFO] Image augmentation process finished.')
    else:
        print('[INFO] Image path csv does not exist, create in data directory.')
        
        
if __name__ == "__main__":
    start_index = 0
    get_data_csv_override(1076)
    main(start_index)
        