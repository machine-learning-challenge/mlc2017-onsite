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

import json
import os
import time

import eval_util
import export_model
import losses
import models
import numpy as np
import random_noise_generator
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/kmlc_gan_train/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "",
      "File glob for the training dataset.")

  flags.DEFINE_string(
      "generator_model", "SampleGenerator",
      "Which architecture to use for the generator model. Models are defined "
      "in models.py.")
  flags.DEFINE_string(
      "discriminator_model", "SampleDiscriminator",
      "Which architecture to use for the discriminator model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
  flags.DEFINE_integer("batch_size", 128,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1.0,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")

  flags.DEFINE_float("base_learning_rate", 0.001,
                     "Which learning rate to start with.")

  flags.DEFINE_float("learning_rate_decay", 0.95,
                     "Learning rate decay factor to be applied every "
                     "learning_rate_decay_examples.")

  flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")

  flags.DEFINE_integer("num_epochs", 50,
                       "How many passes to make over the dataset before "
                       "halting training.")

  flags.DEFINE_integer("max_steps", None,
                       "The maximum number of iterations of the training loop.")

  flags.DEFINE_integer("export_model_steps", 1000,
                       "The period, in number of steps, with which the model "
                       "is exported for batch prediction.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 10,
                       "How many threads to use for reading input files.")

  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")

  flags.DEFINE_float("clip_gradient_norm", 1, "Norm to clip gradients to.")

  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

  flags.DEFINE_bool("export_generated_images", False,
                    "Whether to export the png file including the sample "
                    "generated images during training. Note that it requires "
                    "installing matplotlib.")

  flags.DEFINE_integer("export_image_steps", 500,
                       "The period, in number of steps, with which the "
                       "generated image is exported in png file")

  flags.DEFINE_string("image_dir", "out/",
                      "The directory to save the generated image files in.")

  flags.DEFINE_bool("use_mnist", False,
                    "Whether to use MNIST dataset for easy validation of "
                    "GAN model.")

def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                             (category, flag_value,
                              expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=64,
                           num_epochs=None,
                           num_readers=10):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the ids tensor, images tensor, labels tensor.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def get_gpus():
  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  return gpus

def plot(samples, width, height):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)
  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(height, width), cmap='Greys_r')

  return fig

def build_graph(reader,
                generator_model,
                discriminator_model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
		regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    generator_model: The core model for generator. It should inherit from
                     BaseModel.
    discriminator_model: The core model for discriminator. It should inherit from
                         BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")

  gpus = get_gpus()
  num_gpus = len(gpus)

  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:%d'
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:%d'

  learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step * batch_size * num_towers,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = optimizer_class(learning_rate)

  model_input_raw, _ = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size * num_towers,
          num_readers=num_readers,
          num_epochs=num_epochs))
  tf.summary.histogram("model/input_raw", model_input_raw)
  model_input = model_input_raw

  noise_input = tf.placeholder(
      tf.float32, shape=[None, random_noise_generator.get_dim()])

  image_width, image_height = reader.get_image_size()

  tower_inputs = tf.split(model_input, num_towers)
  tower_noise_input = tf.split(noise_input, num_towers)
  tower_D_gradients = []
  tower_G_gradients = []
  tower_generated_images = []
  tower_predictions_for_fake = []
  tower_predictions_for_real = []
  tower_D_losses = [] 
  tower_G_losses = []

  for i in range(num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
    with tf.device(device_string % i):
      with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
        with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus!=1 else "/gpu:0")):
          generator_model.create_model(image_width * image_height)
          discriminator_model.create_model(image_width * image_height)

          generated_result = generator_model.run_model(tower_noise_input[i])
          generated_images = generated_result["output"]

          generated_images_shaped = tf.reshape(
              generated_images, [-1, image_height, image_width, 1])
          tf.summary.image('generated_images', generated_images_shaped, 10)
          tower_generated_images.append(generated_images)

          result_from_fake = discriminator_model.run_model(generated_images)
          result_from_real = discriminator_model.run_model(tower_inputs[i])
          for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

          predictions_for_fake = result_from_fake["predictions"]
          predictions_for_real = result_from_real["predictions"]
          tower_predictions_for_fake.append(predictions_for_fake)
          tower_predictions_for_real.append(predictions_for_real)

          logits_for_fake = result_from_fake["logits"]
          logits_for_real = result_from_real["logits"]
          D_loss_fake = label_loss_fn.calculate_loss(
              logits_for_fake, tf.zeros_like(logits_for_fake))
          D_loss_real = label_loss_fn.calculate_loss(
              logits_for_real, tf.ones_like(logits_for_real))
          D_loss = D_loss_fake + D_loss_real
          tower_D_losses.append(D_loss)

          G_loss = label_loss_fn.calculate_loss(
              logits_for_fake, tf.ones_like(logits_for_fake))
          tower_G_losses.append(G_loss)

          D_var = discriminator_model.get_variables()
          D_gradients = optimizer.compute_gradients(D_loss, var_list=D_var)
          tower_D_gradients.append(D_gradients)
 
          G_var = generator_model.get_variables()
          G_gradients = optimizer.compute_gradients(G_loss, var_list=G_var)
          tower_G_gradients.append(G_gradients)

  D_loss = tf.reduce_mean(tf.stack(tower_D_losses))
  G_loss = tf.reduce_mean(tf.stack(tower_G_losses))
  tf.summary.scalar("D_loss", D_loss)
  tf.summary.scalar("G_loss", G_loss)
  merged_D_gradients = utils.combine_gradients(tower_D_gradients)
  merged_G_gradients = utils.combine_gradients(tower_G_gradients)

  if clip_gradient_norm > 0:
    with tf.name_scope('clip_grads'):
      merged_D_gradients = utils.clip_gradient_norms(merged_D_gradients, clip_gradient_norm)
      merged_G_gradients = utils.clip_gradient_norms(merged_G_gradients, clip_gradient_norm)

  # Attach global_step only once so that it will be increased by 1.
  D_train_op = optimizer.apply_gradients(merged_D_gradients)
  G_train_op = optimizer.apply_gradients(merged_G_gradients, global_step=global_step)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("D_loss", D_loss)
  tf.add_to_collection("G_loss", G_loss)
  tf.add_to_collection("p_for_fake", tf.concat(tower_predictions_for_fake, 0))
  tf.add_to_collection("p_for_data", tf.concat(tower_predictions_for_real, 0))
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("generated_images", tf.concat(tower_generated_images, 0))
  tf.add_to_collection("D_train_op", D_train_op)
  tf.add_to_collection("G_train_op", G_train_op)
  tf.add_to_collection("noise_input_placeholder", noise_input)

class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, generator_model, discriminator_model, 
               reader, model_exporter, log_device_placement=True, max_steps=None,
               export_model_steps=1000, export_generated_images=False,
               export_image_steps=500, image_dir="out/"):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=log_device_placement)
    self.generator_model = generator_model
    self.discriminator_model = discriminator_model
    self.reader = reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.export_model_steps = export_model_steps
    self.last_model_export_step = 0
    self.export_generated_images = export_generated_images
    self.export_image_steps = export_image_steps
    self.image_dir = image_dir

#     if self.is_master and self.task.index > 0:
#       raise StandardError("%s: Only one replica of master expected",
#                           task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    target, device_fn = self.start_server_if_distributed()

    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    num_towers = max(len(get_gpus()), 1)
    total_batch_size = FLAGS.batch_size * num_towers
    image_width, image_height = self.reader.get_image_size()

    with tf.Graph().as_default() as graph:
      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):
        if not meta_filename:
          saver = self.build_model(
              self.generator_model, self.discriminator_model, self.reader)

        global_step = tf.get_collection("global_step")[0]
        D_loss = tf.get_collection("D_loss")[0]
        G_loss = tf.get_collection("G_loss")[0]
        p_for_fake = tf.get_collection("p_for_fake")[0]
        p_for_real = tf.get_collection("p_for_data")[0]
        generated_images = tf.get_collection("generated_images")[0]

        D_train_op = tf.get_collection("D_train_op")[0]
        G_train_op = tf.get_collection("G_train_op")[0]
        noise_input = tf.get_collection("noise_input_placeholder")[0]
        init_op = tf.global_variables_initializer()
    
    # NOTE: Set save_summaries_sec=0 here because Supervisor doesn't support
    # feeding placeholder on summary_op. Instead, it feeds summary_op manually
    # in below loop.
    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=15 * 60,
        save_summaries_secs=0,
        saver=saver)

    with sv.managed_session(target, config=self.config) as sess:
      try:
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        while (not sv.should_stop()) and (not self.max_steps_reached):
          batch_start_time = time.time()

          noise_input_batch = random_noise_generator.generate_noise(total_batch_size)
          _, _, global_step_val, D_loss_val, G_loss_val, p_fake_val, p_real_val, generated_images_val = sess.run(
              [D_train_op, G_train_op, global_step, D_loss, G_loss, p_for_fake, p_for_real, generated_images],
              feed_dict={noise_input: noise_input_batch})
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = p_real_val.shape[0] / seconds_per_batch

          if self.max_steps and self.max_steps <= global_step_val:
            self.max_steps_reached = True

          if self.is_master and global_step_val % 10 == 0 and self.train_dir:
            eval_start_time = time.time()
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time

            accuracy_on_fake = eval_util.calculate_accuracy_on_fake(p_fake_val)
            accuracy_on_real = eval_util.calculate_accuracy_on_real(p_real_val)

            logging.info("training step " + str(global_step_val) + " | G Loss: " + ("%.4f" % G_loss_val) +
              " | D loss: " + ("%.4f" % D_loss_val) + " | Examples/sec: " + ("%.2f" % examples_per_second) +
              " | D accuracy on G: " + ("%.2f" % accuracy_on_fake) +
              " | D accuracy on real: " + ("%.2f" % accuracy_on_real))

            sv.summary_writer.add_summary(
                utils.MakeSummary("global_step/Examples/Second",
                                  examples_per_second), global_step_val)
            sv.summary_writer.flush()

            # Exporting the model, and gather summary every x steps
            time_to_export = ((self.last_model_export_step == 0) or
                (global_step_val - self.last_model_export_step
                 >= self.export_model_steps))

            if self.is_master and time_to_export:
              self.export_model(global_step_val, sv.saver, sv.save_path, sess)
              self.last_model_export_step = global_step_val
              sv.summary_computed(sess, sess.run(sv.summary_op,
                  feed_dict={noise_input: noise_input_batch}))

          else:
            logging.info("training step " + str(global_step_val) + " | G Loss: " + ("%.4f" % G_loss_val) +
              " | D loss: " + ("%.4f" % D_loss_val) + " | Examples/sec: " + ("%.2f" % examples_per_second))

          # Save some generated image samples in png file.
          if self.is_master and self.export_generated_images and\
              (global_step_val % self.export_image_steps) == 0 and self.image_dir:
            fig = plot(generated_images_val[:16,:], image_width, image_height)
            filename = (self.image_dir + '{}.png').format(
                str(global_step_val / self.export_image_steps).zfill(3))
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            logging.info("Exported image - " + filename)

      except tf.errors.OutOfRangeError:
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()

  def export_model(self, global_step_val, saver, save_path, session):

    # If the model has already been exported at this step, return.
    if global_step_val == self.last_model_export_step:
      return

    last_checkpoint = saver.save(session, save_path, global_step_val)

    model_dir = "{0}/export/step_{1}".format(self.train_dir, global_step_val)
    logging.info("%s: Exporting the model at step %s to %s.",
                 task_as_string(self.task), global_step_val, model_dir)

    self.model_exporter.export_model(
        model_dir=model_dir,
        global_step_val=global_step_val,
        last_checkpoint=last_checkpoint)

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self, generator_model, discriminator_model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    build_graph(reader=reader,
                 generator_model=generator_model,
                 discriminator_model=discriminator_model,
                 optimizer_class=optimizer_class,
                 clip_gradient_norm=FLAGS.clip_gradient_norm,
                 train_data_pattern=FLAGS.train_data_pattern,
                 label_loss_fn=label_loss_fn,
                 base_learning_rate=FLAGS.base_learning_rate,
                 learning_rate_decay=FLAGS.learning_rate_decay,
                 learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                 regularization_penalty=FLAGS.regularization_penalty,
                 num_readers=FLAGS.num_readers,
                 batch_size=FLAGS.batch_size,
                 num_epochs=FLAGS.num_epochs)

    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25,
			  save_relative_paths=True)


def get_reader():
  if FLAGS.use_mnist:
    reader = readers.MnistReader()
  else:
    reader = readers.FaceReader()
  return reader


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    generator_model = find_class_by_name(FLAGS.generator_model, [models])()
    discriminator_model = find_class_by_name(FLAGS.discriminator_model, [models])()

    reader = get_reader()

    model_exporter = export_model.ModelExporter(
        G_model=generator_model,
        D_model=discriminator_model,
        reader=reader)

    Trainer(cluster, task, FLAGS.train_dir, generator_model, discriminator_model,
            reader, model_exporter, FLAGS.log_device_placement, FLAGS.max_steps,
            FLAGS.export_model_steps, FLAGS.export_generated_images, FLAGS.export_image_steps,
            FLAGS.image_dir).run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))

if __name__ == "__main__":
  if FLAGS.export_generated_images:
    exec("""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
if not os.path.exists(FLAGS.image_dir):
  os.makedirs(FLAGS.image_dir)""")      

  app.run()
