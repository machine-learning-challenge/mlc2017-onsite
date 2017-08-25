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

import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import random_noise_generator
import readers

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("G_train_dir", "/tmp/kmlc_gan_train/",
                      "The directory to load the generator model from.")
  flags.DEFINE_string("D_train_dir", "/tmp/kmlc_gan_train_2/",
                      "The directory to load the generator model from.")
  flags.DEFINE_string("output_dir", "result/",
                      "The file to save the generated images to.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. This will be used for real images that will be mixed with "
      "generated images.")
  flags.DEFINE_integer("num_generate", 15,
                       "How many generated images among total output images.")
  flags.DEFINE_integer("num_total_images", 30,
                       "How many images will be used for each round.")

  # Other flags.
  flags.DEFINE_bool("use_mnist", False, "Whether to use MNIST dataset.")
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")


def get_image_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.
  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.
  Returns:
    A image tensor.
  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_ids = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    image_batch, _ = (
        tf.train.batch_join(examples_and_ids,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return image_batch

def _bytes_feature(value):
 return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate(reader, graph, G_train_dir, data_pattern, out_dir, num_generate,
             num_total_images):
  """Generates num_total_images of images in total.
  >> number of generated images : num_generate
  >> number of real image from dataset : (num_total_images - num_generate)

  It outputs following files :
  * ground_truth.csv - a file which has list of (id, labels). Label indicates
    whether it is generated (0) or the real one (1).
  * images/*.png - individual image file. The filename is equal to its id in
    ground_truth.csv."""

  gathered_images = []
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=graph) as sess,\
       gfile.Open(out_dir + "ground_truth.csv", "w+") as out_file:
    latest_checkpoint = tf.train.latest_checkpoint(G_train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % G_train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("Restoring generator model from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)

    real_images_tensor = get_image_data_tensors(reader, data_pattern, 256)
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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    out_file.write("Id,Category\n")

    aggregated_real_images = []

    try:
      while not coord.should_stop():
        real_images = sess.run(real_images_tensor)
        for image in real_images:
          aggregated_real_images.append(image)

    except tf.errors.OutOfRangeError:
      logging.info("Done with reading the test dataset.")
    finally:
      coord.request_stop()

    sampled_real_images = random.sample(
        aggregated_real_images, num_total_images - num_generate)

    generated_images = sess.run(generated_images_tensor,
        feed_dict={noise_input_tensor: random_noise_generator.generate_noise(num_generate)})

    mixed_images = []
    for real in sampled_real_images:
      mixed_images.append([real, 1])
    for generated in generated_images:
      mixed_images.append([generated, 0])
    np.random.shuffle(mixed_images)

    size = len(mixed_images)
    width, height = reader.get_image_size()
    for index in range(size):
      image = mixed_images[index][0]
      label = mixed_images[index][1]

      out_file.write(str(index) + "," + str(label) + "\n")

      gathered_images.append(image)

      image_file_name = (out_dir + 'images/{}.png').format(str(index).zfill(2))
      resized_image_tensor = tf.cast(
          tf.constant(mixed_images[index][0].reshape(height, width, 1)) * 255.,
          tf.uint8)
      encoded_image = sess.run(tf.image.encode_png(resized_image_tensor))
      f = gfile.Open(image_file_name, 'w+')
      f.write(encoded_image)
      f.flush()

      logging.info("Saved %d of %d images.", index + 1, num_total_images)

    out_file.flush()
    logging.info("Saved ground_truth.csv.")

    coord.join(threads)
    sess.close()

  return gathered_images

def discriminate(images, graph, D_train_dir, out_dir):
  """Discriminates images came from generate(), and output the predictions
  as following:
  * predictions.csv - a file which has list of (id, labels). Label indicates
    whether it is generated (0) or the real one (1)."""
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=graph) as sess,\
       gfile.Open(out_dir + "predictions.csv", "w+") as out_file:
    latest_checkpoint = tf.train.latest_checkpoint(D_train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % D_train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("Restoring discriminator model from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)

    input_tensor = tf.get_collection("input_batch_raw")[0]
    predictions_tensor = tf.get_collection("p_for_data")[0]

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

    out_file.write("Id,Category\n")

    predictions = sess.run(predictions_tensor, feed_dict={input_tensor: images})
    one_hot_predictions = 1 * (predictions > 0.5)

    for index in range(len(one_hot_predictions)):
      out_file.write(str(index) + "," + str(one_hot_predictions[index][0]) + "\n")

    out_file.flush()
    logging.info("Saved predictions.csv.")
    sess.close()

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_mnist:
    reader = readers.MnistReader()
  else:
    reader = readers.FaceReader()

  if FLAGS.output_dir is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with generate_and_discriminate.")

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  if not os.path.exists(FLAGS.output_dir + 'images/'):
    os.makedirs(FLAGS.output_dir + 'images/')

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with generate_and_discriminate.")

  if not (0 <= FLAGS.num_generate <= FLAGS.num_total_images):
    raise ValueError("'num_generate' should be between "
      "[0, num_total_images]. Unable to continue "
      "with generate_and_discriminate.")

  # Make separate graphs in order to load two different models.
  G_graph = tf.Graph()
  D_graph = tf.Graph()
  
  images = generate(reader, G_graph, FLAGS.G_train_dir,
                    FLAGS.input_data_pattern, FLAGS.output_dir,
                    FLAGS.num_generate, FLAGS.num_total_images)

  discriminate(images, D_graph, FLAGS.D_train_dir, FLAGS.output_dir)

if __name__ == "__main__":
  app.run()
