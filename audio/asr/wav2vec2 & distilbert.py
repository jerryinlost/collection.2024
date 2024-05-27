import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load a smaller Wav2Vec2 model
model_path = "./models/facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

from transformers import DistilBertForMaskedLM, DistilBertTokenizer, Trainer, TrainingArguments

model_path = "./models/distilbert-base-multilingual-cased"
# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
lm_model = DistilBertForMaskedLM.from_pretrained(model_path)