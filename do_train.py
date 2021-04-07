import os
import random

import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from CyclicLR import CyclicLR
from MultiLabelConfig import get_argparse
from functions import load_device, get_model, convert_examples_to_features, fit
from processor.MultiLabelTextProcessor import MultiLabelTextProcessor as Processor


def get_optimizer(args, model: BertPreTrainedModel, t_total: int, no_decay):
  # Prepare optimizer
  param_optimizer = list(model.named_parameters())
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  if args['local_rank'] != -1:
    t_total = t_total // torch.distributed.get_world_size()
  if args['fp16']:
    try:
      from apex.contrib.optimizers import FP16_Optimizer
      from apex.optimizers import FusedAdam
    except ImportError:
      raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args['learning_rate'],
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args['loss_scale'] == 0:
      optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
      optimizer = FP16_Optimizer(optimizer, static_loss_scale=args['loss_scale'])

  else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args['learning_rate'],
                         warmup=args['warmup_proportion'],
                         t_total=t_total)
  return optimizer


def main(args):
  processor = Processor(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
  device, n_gpu = load_device(args['no_cuda'], args['local_rank'])
  args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])
  seed = args['seed']
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)
  label_list = processor.get_labels()
  num_labels = len(label_list)

  tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])

  train_examples = None
  num_train_steps = None
  if args['do_train']:
    train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    #     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_steps = int(
      len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])
  model = get_model(bert_model=args['bert_model'], num_labels=num_labels, model_state_dict=None)

  if args['fp16']:
    model.half()
  model.to(device)
  if args['local_rank'] != -1:
    try:
      from apex.parallel import DistributedDataParallel as DDP
      model = DDP(model)
    except ImportError:
      raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
  elif n_gpu > 1:
    model = torch.nn.DataParallel(model)
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer = get_optimizer(args, model, num_train_steps, no_decay)

  scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)

  # Eval Fn
  eval_examples = processor.get_dev_examples(args['data_dir'], size=args['val_size'])
  train_features = convert_examples_to_features(
    train_examples, label_list, args['max_seq_length'], tokenizer)
  all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  if args['local_rank'] == -1:
    train_sampler = RandomSampler(train_data)
  else:
    train_sampler = DistributedSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

  fit(model, device, optimizer, n_gpu, train_dataloader, tokenizer, eval_examples, label_list, num_labels, args)

  # Save a trained model
  model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
  output_model_file = os.path.join(args['output_dir'], "finetuned_pytorch_model.bin")
  torch.save(model_to_save.state_dict(), output_model_file)


def to_dict(arg):
  args = {}
  for item in arg._get_kwargs():
    args[item[0]] = item[1]
  return args


if __name__ == '__main__':
  parser = get_argparse()
  args = parser.parse_args()
  main(to_dict(args))
