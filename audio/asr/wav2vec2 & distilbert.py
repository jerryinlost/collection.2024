import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load a smaller Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

from transformers import DistilBertForMaskedLM, DistilBertTokenizer, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
lm_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")