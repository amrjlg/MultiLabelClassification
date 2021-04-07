import os
import pandas as pd

from input import InputExample
from processor.DataProcessor import DataProcessor


class MultiLabelTextProcessor(DataProcessor):

  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.labels = None

  def get_train_examples(self, data_dir, size=-1):
    filename = 'train.csv'

    if size == -1:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
      return self._create_examples(data_df, "train")
    else:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
      return self._create_examples(data_df.sample(size), "train")

  def get_dev_examples(self, data_dir, size=-1):
    """See base class."""
    filename = 'val.csv'
    if size == -1:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
      return self._create_examples(data_df, "dev")
    else:
      data_df = pd.read_csv(os.path.join(data_dir, filename))
      #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
      return self._create_examples(data_df.sample(size), "dev")

  def get_test_examples(self, data_dir, data_file_name, size=-1):
    data_df = pd.read_csv(os.path.join(data_dir, data_file_name))
    #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
    if size == -1:
      return self._create_examples(data_df, "test")
    else:
      return self._create_examples(data_df.sample(size), "test")

  def get_labels(self):
    """See base class."""
    if self.labels == None:
      self.labels = list(pd.read_csv(os.path.join(self.data_dir, "classes.txt"), header=None)[0].values)
    return self.labels

  def _create_examples(self, df, set_type, labels_available=True):
    """Creates examples for the training and dev sets."""
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

