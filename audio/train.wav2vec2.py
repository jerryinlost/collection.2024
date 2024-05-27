from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load a smaller pre-trained model
model_name = "./models/facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)