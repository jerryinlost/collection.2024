from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model and processor
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)