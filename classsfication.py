import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn import BCEWithLogitsLoss


class MultiLabelClassification(BertPreTrainedModel):
  def __init__(self, config, num_labels=2):
    super(MultiLabelClassification, self).__init__(config)
    self.num_labels = num_labels
    self.bert = BertModel(config)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
    self.apply(self.init_bert_weights)

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
    _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    if labels is not None:
      loss_fct = BCEWithLogitsLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
      return loss
    else:
      return logits

  def freeze_bert_encoder(self):
    for param in self.bert.parameters():
      param.requires_grad = False

  def unfreeze_bert_encoder(self):
    for param in self.bert.parameters():
      param.requires_grad = True
