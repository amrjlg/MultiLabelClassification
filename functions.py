import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm

from MultiLabelClassification import MultiLabelClassification
from input.InputFeatures import InputFeatures
from processor.MultiLabelTextProcessor import MultiLabelTextProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
      tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    labels_ids = []
    for label in example.labels:
      labels_ids.append(float(label))

    #         label_id = label_map[example.label]
    if ex_index < 0:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("tokens: %s" % " ".join(
        [str(x) for x in tokens]))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      logger.info(
        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

    features.append(
      InputFeatures(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=labels_ids))
  return features


def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs == labels)


def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = True):
  "Compute accuracy when `y_pred` and `y_true` are the same size."
  if sigmoid: y_pred = y_pred.sigmoid()
  #     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
  return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9,
          sigmoid: bool = True):
  "Computes the f_beta between `preds` and `targets`"
  beta2 = beta ** 2
  if sigmoid: y_pred = y_pred.sigmoid()
  y_pred = (y_pred > thresh).float()
  y_true = y_true.float()
  TP = (y_pred * y_true).sum(dim=1)
  prec = TP / (y_pred.sum(dim=1) + eps)
  rec = TP / (y_true.sum(dim=1) + eps)
  res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
  return res.mean().item()


def warmup_linear(x, warmup=0.002):
  if x < warmup:
    return x / warmup
  return 1.0 - x


# Prepare model
def get_model(model_state_dict, bert_model, num_labels):
  if model_state_dict:
    model = MultiLabelClassification.from_pretrained(bert_model, num_labels=num_labels,
                                                     state_dict=model_state_dict)
  else:
    model = MultiLabelClassification.from_pretrained(bert_model, num_labels=num_labels)
  return model


def eval(model, device, tokenizer, eval_examples, label_list, num_labels, args):
  args['output_dir'].mkdir(exist_ok=True)

  eval_features = convert_examples_to_features(
    eval_examples, label_list, args['max_seq_length'], tokenizer)
  logger.info("***** Running evaluation *****")
  logger.info("  Num examples = %d", len(eval_examples))
  logger.info("  Batch size = %d", args['eval_batch_size'])
  all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
  eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  # Run prediction for full data
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

  all_logits = None
  all_labels = None

  model.eval()
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0
  for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
      tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
      logits = model(input_ids, segment_ids, input_mask)

    #         logits = logits.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         tmp_eval_accuracy = accuracy(logits, label_ids)
    tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
    if all_logits is None:
      all_logits = logits.detach().cpu().numpy()
    else:
      all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    if all_labels is None:
      all_labels = label_ids.detach().cpu().numpy()
    else:
      all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += input_ids.size(0)
    nb_eval_steps += 1

  eval_loss = eval_loss / nb_eval_steps
  eval_accuracy = eval_accuracy / nb_eval_examples

  #     ROC-AUC calcualation
  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()

  for i in range(num_labels):
    fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  result = {'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            #               'loss': tr_loss/nb_tr_steps,
            'roc_auc': roc_auc}

  output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
  with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
      logger.info("  %s = %s", key, str(result[key]))
  #             writer.write("%s = %s\n" % (key, str(result[key])))
  return result


def fit(model, device, optimizer, n_gpu, train_dataloader, tokenizer, eval_examples, label_list, num_labels, args):
  global_step = 0
  model.train()
  num_epocs = args['num_train_epochs']
  for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

      batch = tuple(t.to(device) for t in batch)
      input_ids, input_mask, segment_ids, label_ids = batch
      loss = model(input_ids, segment_ids, input_mask, label_ids)
      if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
      if args['gradient_accumulation_steps'] > 1:
        loss = loss / args['gradient_accumulation_steps']

      if args['fp16']:
        optimizer.backward(loss)
      else:
        loss.backward()

      tr_loss += loss.item()
      nb_tr_examples += input_ids.size(0)
      nb_tr_steps += 1
      if (step + 1) % args['gradient_accumulation_steps'] == 0:
        #             scheduler.batch_step()
        # modify learning rate with special warm up BERT uses
        lr_this_step = args['learning_rate'] * warmup_linear(global_step / t_total, args['warmup_proportion'])
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_this_step
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
    logger.info('Eval after epoc {}'.format(i_ + 1))
    eval(model, device, tokenizer, eval_examples, label_list, num_labels, args)


def predict(model, device, path, label_list, tokenizer, args):
  test_filename = 'test.csv'
  predict_processor = MultiLabelTextProcessor(path)
  test_examples = predict_processor.get_test_examples(path, test_filename, size=-1)

  # Hold input data for returning it
  input_data = [{'id': input_example.guid, 'comment_text': input_example.text_a} for input_example in test_examples]

  test_features = convert_examples_to_features(
    test_examples, label_list, args['max_seq_length'], tokenizer)

  logger.info("***** Running prediction *****")
  logger.info("  Num examples = %d", len(test_examples))
  logger.info("  Batch size = %d", args['eval_batch_size'])

  all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

  test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

  # Run prediction for full data
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

  all_logits = None

  model.eval()
  eval_loss, eval_accuracy = 0, 0
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

  return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=label_list), left_index=True,
                  right_index=True)


def create_trains_csv(path, train_size, dev_size, test_size):
  all = pd.read_csv(os.path.join(path, 'train.csv'))
  train_path = os.path.join(path, 'tmp', "train.csv")
  train = all.sample(train_size)

  write(train_path, train)
  dev_path = os.path.join(path, 'tmp', "dev.csv")
  dev = all.sample(dev_size)
  write(dev_path, dev)
  test_path = os.path.join(path, 'tmp', 'test.csv')
  test = pd.read_csv(os.path.join(path, 'test.csv')).sample(test_size)
  write(test_path, test)


def load_device(no_cuda=True, local_rank=-1):
  if local_rank == -1 or no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
  #     n_gpu = 1
  else:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
  return device, n_gpu


def write(path, datas):
  datas.to_csv(path, index=False)
  # import csv
  #
  # with open(path, 'w', encoding='utf8', newline='') as f:
  #   writer = csv.writer(f)
  #   lines = [datas.columns]
  #   for row in datas.values:
  #     lines.append([str(row[i]) for i in range(len(row))])
  #   writer.writerows(lines)


if __name__ == '__main__':
  # create_trains_csv('/bert_data/jigsaw-toxic-comment-classification-challenge', 5000, 15957, 500)
  # print(len(pd.read_csv('/bert_data/jigsaw-toxic-comment-classification-challenge/train.csv')))
  path = '/root/data/ch/zhihu.json'
  with open(path, 'r', encoding='utf8') as file:
    index = 0
    end = "}\n"
    buffer = ''
    begin = "{\n"
    line = file.readline()
    while line != None:
      if line == begin:
        while line != end:
          value = line.replace('ObjectId(', '').replace('),', ',') \
            .replace('ISODate(', '').replace('),', ',')
          buffer = buffer + value
          line = file.readline()
        buffer = buffer + end
      if line == end:
        break
      line = file.readline()
    print(buffer)
    import json

    s = json.loads(buffer)
    print(s)
