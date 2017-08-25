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

import time

import eval_util
import models
import losses
import random_noise_generator
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/kmlc_gan_train/",
                      "The directory to load the model files from. "
                      "The tensorboard metrics files are also saved to this "
                      "directory.")
  flags.DEFINE_string(
      "eval_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format.")

  flags.DEFINE_string("generator_model", "SampleGenerator",
                      "Which generator model to use: see models.py")

  flags.DEFINE_string("discriminator_model", "SampleDiscriminator",
                      "Which discriminator odel to use: see models.py")

  flags.DEFINE_integer("batch_size", 64,
                       "How many examples to process per batch.")

  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Loss computed on validation data")

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")

  flags.DEFINE_boolean("run_once", True, "Whether to run eval only once.")

  flags.DEFINE_bool("use_mnist", False,
                    "Whether to use MNIST dataset for easy validation of "
                    "GAN model.")

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=64,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the images and labels

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.batch_join(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(reader,
                generator_model,
                discriminator_model,
                eval_data_pattern,
                label_loss_fn,
                batch_size=128,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  model_input_raw, labels_batch = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
      reader,
      eval_data_pattern,
      batch_size=batch_size)
  tf.summary.histogram("model_input_raw", model_input_raw)

  model_input = model_input_raw
  noise_input = tf.placeholder(
      tf.float32, shape=[None, random_noise_generator.get_dim()])

  with tf.variable_scope("tower"):
    image_width, image_height = reader.get_image_size()
    generator_model.create_model(image_width * image_height)
    discriminator_model.create_model(image_width * image_height)

    generated_images = generator_model.run_model(
        noise_input, is_training=False)["output"]

    generated_images_shaped = tf.reshape(
        generated_images, [-1, image_height, image_width, 1])
    tf.summary.image('generated_images', generated_images_shaped, 10)

    result_from_fake = discriminator_model.run_model(
        generated_images, is_training=False)
    result_from_real = discriminator_model.run_model(
        model_input, is_training=False)

    predictions_for_fake = result_from_fake["predictions"]
    predictions_for_real = result_from_real["predictions"]
    tf.summary.histogram("D_predictions_for_real", predictions_for_real)
    tf.summary.histogram("D_predictions_for_fake", predictions_for_fake)

    logits_for_fake = result_from_fake["logits"]
    logits_for_real = result_from_real["logits"]
    D_loss_fake = label_loss_fn.calculate_loss(
        logits_for_fake, tf.zeros_like(logits_for_fake))
    D_loss_real = label_loss_fn.calculate_loss(
        logits_for_real, tf.ones_like(logits_for_real))
    D_loss = tf.cast(D_loss_fake, tf.float32) + tf.cast(D_loss_real, tf.float32)

    G_loss = label_loss_fn.calculate_loss(
        logits_for_fake, tf.ones_like(logits_for_fake))

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("D_loss", D_loss)
  tf.add_to_collection("G_loss", G_loss)
  tf.add_to_collection("p_for_fake", predictions_for_fake)
  tf.add_to_collection("p_for_data", predictions_for_real)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("summary_op", tf.summary.merge_all())
  tf.add_to_collection("noise_input_placeholder", noise_input)


def evaluation_loop(p_fake_batch, p_real_batch, G_loss, D_loss, noise_input,
                    summary_op, saver, summary_writer, evl_metrics,
                    last_global_step_val, batch_size):
  """Run the evaluation loop once.

  Args:
    p_fake_batch: a tensor of predictions mini-batch for fake images.
    p_real_batch: a tensor of predictions mini-batch for real images.
    G_loss: a tensor of generator loss for the examples in the mini-batch.
    D_loss: a tensor of discriminator loss for the examples in the mini-batch.
    noise_input: a placeholder tensor which holds random noise for generator.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.
    batch_size: a size for batch.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    if latest_checkpoint:
      logging.info("Loading checkpoint for eval: " + latest_checkpoint)
      # Restores from checkpoint
      saver.restore(sess, latest_checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = latest_checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info("skip this checkpoint global_step_val=%s "
                   "(same as the previous one).", global_step_val)
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [p_fake_batch, p_real_batch, G_loss, D_loss, summary_op]
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()
        noise_input_batch = random_noise_generator.generate_noise(batch_size)
        p_fake_val, p_real_val, G_loss_val, D_loss_val, summary_val = sess.run(
            fetches, feed_dict={noise_input: noise_input_batch})
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = p_real_val.shape[0] / seconds_per_batch
        examples_processed += p_real_val.shape[0]

        iteration_info_dict = evl_metrics.accumulate(p_fake_val, p_real_val,
                                                     G_loss_val, D_loss_val)
        iteration_info_dict["examples_per_second"] = example_per_second

        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def evaluate():
  tf.set_random_seed(0)  # for reproducibility
  with tf.Graph().as_default():
    if FLAGS.use_mnist:
      reader = readers.MnistReader()
    else:
      reader = readers.FaceReader()

    generator_model = find_class_by_name(FLAGS.generator_model, [models])()
    discriminator_model = find_class_by_name(FLAGS.discriminator_model, [models])()
    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()

    if FLAGS.eval_data_pattern is "":
      raise IOError("'eval_data_pattern' was not specified. " +
                     "Nothing to evaluate.")

    build_graph(
        reader=reader,
        generator_model=generator_model,
        discriminator_model=discriminator_model,
        eval_data_pattern=FLAGS.eval_data_pattern,
        label_loss_fn=label_loss_fn,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size)
    logging.info("built evaluation graph")
    p_fake_batch = tf.get_collection("p_for_fake")[0]
    p_real_batch = tf.get_collection("p_for_data")[0]
    G_loss = tf.get_collection("G_loss")[0]
    D_loss = tf.get_collection("D_loss")[0]
    noise_input = tf.get_collection("noise_input_placeholder")[0]
    summary_op = tf.get_collection("summary_op")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir, graph=tf.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics()

    last_global_step_val = -1
    while True:
      last_global_step_val = evaluation_loop(p_fake_batch, p_real_batch,
                                             G_loss, D_loss, noise_input,
                                             summary_op, saver, summary_writer,
                                             evl_metrics, last_global_step_val,
                                             FLAGS.batch_size)
      if FLAGS.run_once:
        break


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  evaluate()


if __name__ == "__main__":
  app.run()
