# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging

import random_noise_generator
import readers

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("train_dir", "/tmp/kmlc_gan_train/",
                      "The directory to load the model files from.")
  flags.DEFINE_string("output_file", "interpolation.png",
                      "The file to save the generated images to.")

  # Other flags.
  flags.DEFINE_bool("use_mnist", False, "Whether to use MNIST dataset.")

def save_image(image, file_name):
  fig = plt.figure()
  ax = plt.Axes(fig, [0, 0, 1, 1])
  ax.set_axis_off()
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_aspect('equal')
  fig.add_axes(ax)      
  ax.imshow(image, cmap='Greys_r')
  plt.savefig(file_name, bbox_inches='tight')
  plt.close()

def _bytes_feature(value):
 return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def linear_interpolation(reader, train_dir, output_file):
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)

    noise_input_tensor = tf.get_collection("noise_input_placeholder")[0]
    generated_images_tensor = tf.get_collection("generated_images")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    noise_square = random_noise_generator.generate_noise(4)
    noise_linear_interpolation = []

    for i in range(10):
      for j in range(10):
        fi = float(i) / 9.
        fj = float(j) / 9.
        topline = noise_square[0] * (1 - fi) + noise_square[1] * fi
        botline = noise_square[2] * (1 - fi) + noise_square[3] * fi
        noise_linear_interpolation.append(topline * (1 - fj) + botline * fj)

    generated_images = sess.run(generated_images_tensor,
        feed_dict={noise_input_tensor: noise_linear_interpolation})

    width, height = reader.get_image_size()

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(generated_images):
      ax = plt.subplot(gs[i])
      plt.axis('off')
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_aspect('equal')
      plt.imshow(sample.reshape(height, width), cmap='Greys_r')

    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    sess.close()

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_mnist:
    reader = readers.MnistReader()
  else:
    reader = readers.FaceReader()

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with linear_interpolation.")

  linear_interpolation(reader, FLAGS.train_dir, FLAGS.output_file)


if __name__ == "__main__":
  app.run()
