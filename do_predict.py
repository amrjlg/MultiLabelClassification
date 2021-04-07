from collections import Iterator

import numpy as np
import pandas as pd
import torch
from datasets import tqdm
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset

from MultiLabelConfig import defaultMultiLabelConfig
from classsfication import MultiLabelClassification
from functions import load_device, convert_examples_to_features
from input import InputExample
from processor.MultiLabelTextProcessor import MultiLabelTextProcessor


def to_dict(arg):
  args = {}
  for item in arg._get_kwargs():
    args[item[0]] = item[1]
  return args


def predict(model: MultiLabelClassification, test_examples: Iterator[InputExample], label_list):
  # Hold input data for returning it
  input_data = [{'id': input_example.guid, 'comment_text': input_example.text_a} for input_example in test_examples]
  tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])
  test_features = convert_examples_to_features(
    test_examples, label_list, args['max_seq_length'], tokenizer)

  all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

  test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

  # Run prediction for full data
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

  all_logits = None

  model.eval()
  device = load_device()
  nb_eval_steps, nb_eval_examples = 0, 0
  for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
    input_ids, input_mask, segment_ids = batch
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    with torch.no_grad():
      logits = model(input_ids, segment_ids, input_mask)
      logits = logits.sigmoid()

    if all_logits is None:
      all_logits = logits.detach().cpu().numpy()
    else:
      all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    nb_eval_examples += input_ids.size(0)
    nb_eval_steps += 1

  return pd.merge(
    pd.DataFrame(input_data),
    pd.DataFrame(all_logits, columns=label_list),
    left_index=True,
    right_index=True)


def main(args):
  processor = MultiLabelTextProcessor(args['data_path'])
  model_state_dict = torch.load(args['fine_path'])
  label_list = processor.get_labels()
  model = MultiLabelClassification.from_pretrained(args['bert_model'], num_labels=len(label_list),
                                                   state_dict=model_state_dict)
  device = load_device()
  model.to(device)

  result = predict(model, processor.get_test_examples(args['data_path'], args['file_name']), processor.get_labels())

  print(result.shape)

  DATA_PATH = args['data_path']
  result['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'].to_csv(
    DATA_PATH / 'toxic_comment_text.csv', index=None)
  cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  result[cols].to_csv(DATA_PATH / 'toxic_labels.csv', index=None)


if __name__ == '__main__':
  import argparse

  parse = argparse.ArgumentParser(description='args info')
  parse.add_argument("-bert_model", default=defaultMultiLabelConfig['bert_model'], help='bert model path')
  parse.add_argument("-fine_path", default=defaultMultiLabelConfig['output_dir'], help='trained model path')
  parse.add_argument("-data_path", default=defaultMultiLabelConfig['data_dir'], help='validate file path')
  parse.add_argument("-file_name", default='test.csv', help='validate file name')
  parse.add_argument("-out_path", default=defaultMultiLabelConfig['output_dir'], help='path to save result')
  parse.add_argument("-out_file", default='predict.csv', help='saved file name')
  args = parse.parse_args()
  main(to_dict(args))
