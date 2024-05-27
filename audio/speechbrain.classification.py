import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="./models/speechbrain/google_speech_command_xvector")
out_prob, score, index, text_lab = classifier.classify_file('speechbrain/google_speech_command_xvector/yes.wav')
print(text_lab)
out_prob, score, index, text_lab = classifier.classify_file('speechbrain/google_speech_command_xvector/stop.wav')
print(text_lab)

# cd recipes/Google-speech-commands
# python train.py hparams/xvect.yaml --data_folder=your_data_folder
