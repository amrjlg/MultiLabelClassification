defaultMultiLabelConfig = {
  "bert_model": './bert',
  "output_dir": './out',
  "data_dir": './data',
  "train_size": -1,
  "dev_size": -1,
  "test_size": -1,
  "no_cuda": False,
  "do_train": True,
  "do_eval": True,
  "do_lower_case": True,
  "max_seq_length": 512,
  "train_batch_size": 32,
  "eval_batch_size": 32,
  "learning_rate": 3e-5,
  "num_train_epochs": 4.0,
  "warmup_proportion": 0.1,
  "local_rank": -1,
  "seed": 42,
  "gradient_accumulation_steps": 1,
  "optimize_on_cpu": False,
  "fp16": False,
  "loss_scale": 128
}


def get_argparse():
  import argparse
  parse = argparse.ArgumentParser(description='args info')
  for key in defaultMultiLabelConfig:
    parse.add_argument('--' + key, default=defaultMultiLabelConfig[key])
  return parse


if __name__ == '__main__':
  parse = get_argparse()
  args = parse.parse_args()
  print(args)
