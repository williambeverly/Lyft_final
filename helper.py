import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

         # generate paths for images and labels from multiple paths
        image_paths = []
        label_paths = []
        delimiter = '/'

        for pathname in os.walk(data_folder):
            if(os.path.basename(pathname[0]) == 'CameraRGB'):
                for image in pathname[2]:
                    image_paths.append(os.path.join(pathname[0], image))
            if(os.path.basename(pathname[0]) == 'CameraSeg'):
                for image in pathname[2]:
                    label_paths.append(os.path.join(pathname[0], image))

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                image_file_path = image_file.split(delimiter)
                image_file_path[-2] = 'CameraSeg'
                gt_image_file = delimiter.join(image_file_path)

                image = scipy.misc.imread(image_file)
                image = image[15:495,:,:]
                
                image = scipy.misc.imresize(image, image_shape)

                gt_image = scipy.misc.imread(gt_image_file)
                gt_image = gt_image[15:495,:,:]
                gt_image = scipy.misc.imresize(gt_image, image_shape)

                gt_image = gt_image[:,:,0] # take the red channel
                gt_bg = (gt_image == 7) + (gt_image == 6)
                gt_vehicle = gt_image == 10
                gt_other = np.invert(gt_bg + gt_vehicle)
                
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_vehicle = gt_vehicle.reshape(*gt_vehicle.shape, 1)
                gt_other = gt_other.reshape(*gt_other.shape, 1)

                gt_image = np.concatenate((gt_bg, gt_vehicle, gt_other), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, prediction, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    crop_img_shp = (480, 800)

    for image_file in glob(os.path.join(data_folder, '*.png')):
        #image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        image = scipy.misc.imread(image_file)
        image = image[15:495,:,:]

        image = scipy.misc.imresize(image, image_shape)

        pred = sess.run(prediction, feed_dict={image_input: [image], keep_prob:1.0}) # get the prediction

        # reshape the prediction
        pred = pred.reshape(image_shape)

        # get the road and the vehicle
        road = (pred == 0).astype('uint8')
        vehicle = (pred == 1).astype('uint8')
        
        # resize to final shape
        road = scipy.misc.imresize(road, crop_img_shp)
        vehicle = scipy.misc.imresize(vehicle, crop_img_shp)
        
        # stack some space on each to make up the final image
        road = np.vstack((np.zeros((15, 800), 'uint8'), road, np.zeros((105, 800), 'uint8')))
        vehicle = np.vstack((np.zeros((15, 800), 'uint8'), vehicle, np.zeros((105, 800), 'uint8')))

        # create the final image
        final_image = np.zeros((600, 800, 3), 'uint8')
        final_image[:,:,0] = road
        final_image[:,:,1] = vehicle

        street_im = scipy.misc.toimage(final_image)
        
        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, prediction, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, prediction, keep_prob, input_image, os.path.join(data_dir, 'video'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
