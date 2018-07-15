import numpy as np
from keras.preprocessing import image as image_processor
from keras.utils import np_utils
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

#size of image, inceptionv3 is 299x299, resnet50 is 224x224
img_size = 299
#number of cpus to use for image processing, i choose 4, should run for most people. basically num of cores your machine has
proccesses_num = 4

#function that loads in a specific dataset and creates the labels
def load_image_path_set(set_type):
    path_melanoma = 'data/' + set_type + '/melanoma/*'
    path_nevus = 'data/' + set_type + '/nevus/*'
    path_seb = 'data/' + set_type + '/seborrheic_keratosis/*'

    data_melanoma = np.array(glob(path_melanoma))
    data_nevus = np.array(glob(path_nevus))
    data_seb = np.array(glob(path_seb))
    data = np.concatenate((data_melanoma, data_nevus, data_seb))

    #melanoma labels, class 0
    labels = np.full(data_melanoma.shape, 0)
    #append labels with nevus labels, class 1
    labels = np.append(labels, np.full(data_nevus.shape, 1))
    #append labels with seb labels, class 2
    labels = np.append(labels, np.full(data_seb.shape, 2))

    #shuffle dataset since they are in order of diease, this generates randomized indicies of data
    randomize_indicies = np.arange(data.shape[0])
    np.random.shuffle(randomize_indicies)

    #one-hot encode labels:
    labels = np_utils.to_categorical(labels, num_classes=3)

    return data[randomize_indicies], labels[randomize_indicies]

#processes single image to test in test_model.py. Need to expand dimensions so to get a tensor with shape (1, img_size, img_size, 3)
def process_single_image(img_path):
    img = image_processor.load_img(img_path, target_size=(img_size,img_size))
    return np.expand_dims(image_processor.img_to_array(img), axis=0)

#takes an array of image paths and returns list of values of image in a 3D tensor (num_of_images, img_size, img_size, 3)
def image_processor_job(img_paths_short):

    images_short = []
    for pic in tqdm(img_paths_short):
        img = image_processor.load_img(pic, target_size=(img_size,img_size))
        img_arr = image_processor.img_to_array(img)
        images_short.append(img_arr)
    return images_short

#loads images from image paths, resizes them, and then turns them into numpy arrays.
#use multiple processes since image conversion takes a while on a single core.
def load_images(img_paths):
    pool = Pool(processes=proccesses_num)

    split_size = len(img_paths)//proccesses_num

    #split data into processes_num batches
    batches = []
    for i in range(0, proccesses_num*split_size, split_size):
        #if not on last batch, batch is split size
        if i != (proccesses_num-1)*split_size:
            batches.append([img_paths[i:i+split_size]])
        else: #batch is rest of img paths
            batches.append([img_paths[i:]])
    print("Starting image processing on multiple threads...")
    results = np.vstack(pool.starmap(image_processor_job, batches))
    return results

#saves images so don't have to keep processing the images each run.
def save_images_array(images, filename):
    dir = 'data/' + filename
    np.save(dir, images)

#load data from certain set as read only
def load_images_and_labels_array(set_type):
    images, targets = 'data/' + set_type + '_images.npy', 'data/' + set_type +'_targets.npy'
    return np.load(images, mmap_mode='r'), np.load(targets, mmap_mode='r')
