from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante."
model = TTS(language='FR', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'fr.wav'
model.tts_to_file(text, speaker_ids['FR'], output_path, speed=speed)