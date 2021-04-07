import os
import pandas as pd

from input.InputExample import InputExample
from processor.DataProcessor import DataProcessor


class MultiLabelTextProcessor(DataProcessor):

  def __init__(self, labels):
    self.labels = labels

  def get_train_examples(self, data_dir, size=-1):
    filename = 'train.csv'

    if size == -1:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      return self._create_examples(data_df)
    else:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      return self._create_examples(data_df.sample(size))

  def get_dev_examples(self, data_dir, size=-1):
    filename = 'val.csv'
    if size == -1:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      return self._create_examples(data_df)
    else:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      return self._create_examples(data_df.sample(size))

  def get_test_examples(self, data_dir, data_file_name, size=-1):
    data_df = pd.read_csv(os.path.join(data_dir, data_file_name))
    if size == -1:
      return self._create_examples(data_df)
    else:
      return self._create_examples(data_df.sample(size))

  def get_labels(self):
    return self.labels

  def _create_examples(self, df, labels_available=True):
    examples = []
    for (i, row) in enumerate(df.values):
      guid = row[0]
      text_a = row[1]
      if labels_available:
        labels = row[2:]
      else:
        labels = []
      examples.append(
        InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples
