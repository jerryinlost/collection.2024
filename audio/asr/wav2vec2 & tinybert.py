import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")