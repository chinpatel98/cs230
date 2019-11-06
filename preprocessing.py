from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

import pathlib

# Import the images into the file
str_data_dir = './images/Train/'
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_dir = pathlib.Path(str_data_dir)
image_count = len(list(data_dir.glob('*.jpg')))
print("There are {} images".format(image_count))

BATCH_SIZE = 16
IMG_HEIGHT = 1080
IMG_WIDTH = 1080
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

def crop_center_and_resize(image, resize_height, resize_width):
  h, w = image.shape[-3], image.shape[-2]
  if h > w:
      cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
  else:
      cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
  return tf.image.resize_images(cropped_image, (resize_height, resize_width))

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)

  # resize the image to the desired size.
  return tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=0, target_height=IMG_HEIGHT, target_width=IMG_WIDTH)
  #return crop_center_and_resize(img, IMG_HEIGHT, IMG_WIDTH)


def process_path(file_path):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  #ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


def get_train_ds():
  # Preprocess them
  list_dataset = tf.data.Dataset.list_files(str_data_dir + '/*.jpg')

  # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
  print('Loading dataset')
  loaded_images_dataset = list_dataset.map(process_path, num_parallel_calls=AUTOTUNE)

  print('Preparing dataset for training')
  train_dataset = prepare_for_training(loaded_images_dataset)
  return train_dataset


def show_batch(image_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.axis('off')
    plt.show()

train_dataset = get_train_ds()  

# for blurred_batch in train_dataset:
#   show_batch(blurred_batch.numpy())
#   break

count = 0
for batch in train_dataset:
  print("Batch #{}".format(count))
  count += 1



################# Tutorials #################


# https://www.tensorflow.org/tutorials/load_data/images


