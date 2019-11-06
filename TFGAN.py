### Import TensorFlow and other libraries

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers, Model
import tensorlayer as tl
import time
import pathlib

"""### Load and prepare the dataset

You will use the MNIST dataset to train the generator and the discriminator. The generator will generate handwritten digits resembling the MNIST data.
"""

from preprocessing import train_dataset

# Import the images into the file
str_data_dir = './images/Train/'
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_dir = pathlib.Path(str_data_dir)
image_count = len(list(data_dir.glob('*.jpg')))
print("There are {} images".format(image_count))

BATCH_SIZE = 16
HR_IMG_HEIGHT = 1080
HR_IMG_WIDTH = 1080

DOWNSAMPLING_FACTOR = 4

LR_IMG_HEIGHT = HR_IMG_HEIGHT/DOWNSAMPLING_FACTOR
LR_IMG_WIDTH = HR_IMG_WIDTH/DOWNSAMPLING_FACTOR

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
NUM_CHANNELS = 3

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16


def generate_gaussian_kernel(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



"""## Create the models

Both the generator and discriminator are defined using the [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

### The Generator

The generator uses `tf.keras.layers.Conv2DTranspose` (upsampling) layers to produce an image from a seed (random noise). Start with a `Dense` layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the `tf.keras.layers.LeakyReLU` activation for each layer, except the output layer which uses tanh.
"""

B = 16 # Number of generator residual blocks

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
# Implementation by Shi et al. (https://github.com/twairball/keras-subpixel-conv)
from subpixel import SubpixelConv2D, Subpixel



############################################# BLUR AND DOWNSAMPLE LAYERS #############################################

# Gaussian Blur Setup
BLUR_KERNEL_SIZE = 3
kernel_weights = generate_gaussian_kernel()

# Size compatibility code
kernel_weights = np.expand_dims(kernel_weights, axis=-1)
kernel_weights = np.repeat(kernel_weights, NUM_CHANNELS, axis=-1) # apply the same filter on all the input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons

# Blur
blur_layer = layers.DepthwiseConv2D(BLUR_KERNEL_SIZE, use_bias=False, padding='same')

# Downsample
downsample_layer = layers.AveragePooling2D(pool_size=(DOWNSAMPLING_FACTOR, DOWNSAMPLING_FACTOR))

############################################# BLUR AND DOWNSAMPLE LAYERS #############################################


################################################### MODEL CREATION ###################################################

def make_downsampler_model():
    hr_img = layers.Input(shape=(HR_IMG_WIDTH, HR_IMG_HEIGHT, NUM_CHANNELS))

    lr_img = blur_layer(hr_img)
    lr_img = downsample_layer(lr_img)

    return Model(inputs=hr_img, outputs=lr_img, name='downsampler')

downsampler = make_downsampler_model()
blur_layer.set_weights([kernel_weights])
blur_layer.trainable = False  # the weights should not change during training

def make_sr_generator_model():
    lr_img = layers.Input(shape=(LR_IMG_WIDTH, LR_IMG_HEIGHT, NUM_CHANNELS))

    ################################################################################
    ## Now that we have a low res image, we can start the actual generator ResNet ##
    ################################################################################

    x = layers.Convolution2D(64, (9,9), (1,1), padding='same')(lr_img)
    x = layers.PReLU()(x)

    b_prev = x


    #####################
    ## Residual Blocks ##
    #####################

    for i in range(B):
      b_curr = layers.Convolution2D(64, (3,3), (1,1), padding='same')(b_prev)
      b_curr = layers.BatchNormalization()(b_curr)
      b_curr = layers.PReLU()(b_curr)
      b_curr = layers.Convolution2D(64, (3,3), (1,1), padding='same')(b_curr)
      b_curr = layers.BatchNormalization()(b_curr)
      b_curr = layers.Add()([b_prev, b_curr]) #skip connection

      b_prev = b_curr

    res_out = b_curr # Output of residual blocks

    x2 = layers.Convolution2D(64, (3,3), (1,1), padding='same')(res_out)
    x2 = layers.BatchNormalization()(x2)
    x = layers.Add()([x, x2]) #skip connection


    #######################################################
    ## Resolution-enhancing sub-pixel convolution layers ##
    #######################################################

    # Layer 1 (Half of the upsampling)
    x = layers.Convolution2D(256, (3,3), (1,1), padding='same')(res_out)
    x = SubpixelConv2D(input_shape=(None, LR_IMG_WIDTH, LR_IMG_HEIGHT, NUM_CHANNELS), scale=DOWNSAMPLING_FACTOR/2, idx=0)(x)
    #x = Subpixel(256, kernel_size=(3,3), r=DOWNSAMPLING_FACTOR/2, padding='same', strides=(1,1))
    x = layers.PReLU()(x)

    # Layer 2 (Second half of the upsampling)
    x = layers.Convolution2D(256, (3,3), (1,1), padding='same')(x)
    x = SubpixelConv2D(input_shape=(None, LR_IMG_WIDTH*(DOWNSAMPLING_FACTOR/2), LR_IMG_HEIGHT*(DOWNSAMPLING_FACTOR/2), NUM_CHANNELS/((DOWNSAMPLING_FACTOR/2) ** 2)), scale=(DOWNSAMPLING_FACTOR/2), idx=1)(x)
    #x = Subpixel(256, kernel_size=(3,3), r=DOWNSAMPLING_FACTOR/2, padding='same', strides=(1,1))
    x = layers.PReLU()(x)

    generated_sr_image = layers.Convolution2D(3, (9,9), (1,1), padding='same')(x)
    output_shape = generated_sr_image.get_shape().as_list()
    assert output_shape == [None, HR_IMG_HEIGHT, HR_IMG_WIDTH, NUM_CHANNELS]

    return Model(inputs=lr_img, outputs=generated_sr_image, name='generator')

generator = make_sr_generator_model()


def make_sr_discriminator_model():
    inputs = layers.Input(shape=(HR_IMG_HEIGHT, HR_IMG_WIDTH, NUM_CHANNELS))
    # k3n64s1
    x = layers.Convolution2D(64, (3,3), (1,1), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #################
    ## Conv Blocks ##
    #################

    # k3n64s2
    x = layers.Convolution2D(64, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n128s1
    x = layers.Convolution2D(128, (3,3), (1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n128s2
    x = layers.Convolution2D(128, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n256s1
    x = layers.Convolution2D(256, (3,3), (1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n256s2
    x = layers.Convolution2D(256, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n512s1
    x = layers.Convolution2D(512, (3,3), (1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n512s2
    x = layers.Convolution2D(512, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    ################
    ## Dense Tail ##
    ################

    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs, name='discriminator')


discriminator = make_sr_discriminator_model()

################################################### MODEL CREATION ###################################################

################################################### LOSS AND OPTIMIZER ###############################################

"""## Define the loss and optimizers

Define loss functions and optimizers for both models.
"""

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""### Discriminator loss

This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
"""

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

"""### Generator loss
The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.
"""

def generator_loss(generated_images, hr_images, real_output=None, fake_output=None, VGG=None, pretraining=False):
    if pretraining:
      return tl.cost.mean_squared_error(generated_images, hr_images, is_mean=True)
    else:
      fake_image_features = VGG((generated_images+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
      real_image_features = VGG((hr_images+1)/2.)

      g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(fake_output, tf.ones_like(fake_output))
      mse_loss = tl.cost.mean_squared_error(generated_images, hr_images, is_mean=True)
      vgg_loss = 2e-6 * tl.cost.mean_squared_error(fake_image_features, real_image_features, is_mean=True)
      g_loss = mse_loss + vgg_loss + g_gan_loss
      return g_loss

"""The discriminator and the generator optimizers are different since we will train two networks separately."""

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

"""### Save checkpoints
This notebook also demonstrates how to save and restore models, which can be helpful in case a long running training task is interrupted.
"""

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

################################################### LOSS AND OPTIMIZER ###############################################

###################################################### TRAINING LOOP #################################################

"""The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator."""

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, VGG):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      lr_images = downsampler(images, training=False)
      generated_images = generator(lr_images, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      disc_loss = discriminator_loss(real_output, fake_output)
      gen_loss = generator_loss(generated_images, images, real_output, fake_output, VGG)
      

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, test_dataset=None):
  VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')
  VGG.eval()

  for epoch in range(epochs):
    start = time.time()

    count = 0
    for image_batch in dataset:
      #print("Batch {}".format(count))
      train_step(image_batch, VGG)
      count += 1

    # Produce images for the GIF as we go
    if test_dataset:
      generate_and_save_images(generator,
                               epoch + 1,
                               test_dataset)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {}/{} is {} sec'.format(epoch + 1, epochs, time.time()-start))

  # Generate after the final epoch
  if test_dataset:
    generate_and_save_images(generator,
                             epochs,
                             test_dataset)

"""**Generate and save images**"""

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :])
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  if (epoch % 50 == 0):
    plt.show()

###################################################### TRAINING LOOP #################################################

"""## Train the model
Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
train(train_dataset, EPOCHS)

"""Restore the latest checkpoint."""

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Create a GIF"""

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

"""Use `imageio` to create an animated gif using the images saved during training."""

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)