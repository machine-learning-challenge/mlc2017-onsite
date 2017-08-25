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

"""Provides functions to help with evaluating models."""
import datetime
import numpy as np

from tensorflow.python.platform import gfile

def calculate_accuracy_on_fake(predictions):
  return calculate_accuracy(predictions, np.zeros_like(predictions))

def calculate_accuracy_on_real(predictions):
  return calculate_accuracy(predictions, np.ones_like(predictions))

def calculate_accuracy(predictions, actuals):
  """Performs a local (numpy) calculation of the accuracy.
  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x '1'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x '1'.
  Returns:
    float: The average hit at one across the entire batch.
  """
  one_hot_predictions = 1.0 * (predictions > 0.5)
  return np.sum(1.0 * (one_hot_predictions == actuals)) / len(predictions)

class EvaluationMetrics(object):
  """A class to store the evaluation metrics."""

  def __init__(self):
    """Construct an EvaluationMetrics object to store the evaluation metrics.

    Args:
      num_class: A positive integer specifying the number of classes.

    Raises:
      ValueError: An error occurred when MeanAveragePrecisionCalculator cannot
        not be constructed.
    """
    self.sum_accuracy_on_fake = 0.0
    self.sum_accuracy_on_real = 0.0
    self.sum_G_loss = 0.0
    self.sum_D_loss = 0.0
    self.num_examples = 0

  def accumulate(self, p_for_fake, p_for_real, G_loss, D_loss):
    """Accumulate the metrics calculated locally for this mini-batch.

    Args:
      p_for_fake: A numpy matrix containing the outputs of the model on fake
        (generated) images. Dimensions are 'batch' x 1.
      p_for_real: A numpy matrix containing the outputs of the model on real
        (data set) images. Dimensions are 'batch' x 1.
      G_loss: A numpy array containing the loss of generator model for each
        sample.
      G_loss: A numpy array containing the loss of discriminator for each
        sample.

    Returns:
      dictionary: A dictionary storing the metrics for the mini-batch.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
        does not match.
    """
    batch_size = p_for_real.shape[0]
    mean_accuracy_on_fake = calculate_accuracy_on_fake(p_for_fake)
    mean_accuracy_on_real = calculate_accuracy_on_real(p_for_real)
    mean_G_loss = np.mean(G_loss)
    mean_D_loss = np.mean(D_loss)

    self.num_examples += batch_size
    self.sum_accuracy_on_fake += mean_accuracy_on_fake * batch_size
    self.sum_accuracy_on_real += mean_accuracy_on_real * batch_size
    self.sum_G_loss += mean_G_loss * batch_size
    self.sum_D_loss += mean_D_loss * batch_size

    return {"accuracy_on_fake": mean_accuracy_on_fake,
            "accuracy_on_real": mean_accuracy_on_real,
            "G_loss": mean_G_loss, "D_loss": mean_D_loss}

  def get(self):
    """Calculate the evaluation metrics for the whole epoch.

    Raises:
      ValueError: If no examples were accumulated.

    Returns:
      dictionary: a dictionary storing the evaluation metrics for the epoch. The
        dictionary has the fields: avg_hit_at_one, avg_perr, avg_loss, and
        aps (default nan).
    """
    if self.num_examples <= 0:
      raise ValueError("total_sample must be positive.")
    avg_accuracy_on_fake = self.sum_accuracy_on_fake / self.num_examples
    avg_accuracy_on_real = self.sum_accuracy_on_real / self.num_examples
    avg_G_loss = self.sum_G_loss / self.num_examples
    avg_D_loss = self.sum_D_loss / self.num_examples

    return {"avg_accuracy_on_fake": avg_accuracy_on_fake,
            "avg_accuracy_on_real": avg_accuracy_on_real,
            "avg_G_loss": avg_G_loss, "avg_D_loss": avg_D_loss}

  def clear(self):
    """Clear the evaluation metrics and reset the EvaluationMetrics object."""
    self.sum_accuracy_on_fake = 0.0
    self.sum_accuracy_on_real = 0.0
    self.sum_G_loss = 0.0
    self.sum_D_loss = 0.0
    self.num_examples = 0
