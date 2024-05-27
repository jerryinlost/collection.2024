from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="./models/speechbrain/sepformer-dns4-16k-enhancement")

# for custom file, change path
est_sources = model.separate_file(path='speechbrain/sepformer-dns4-16k-enhancement/example_dns4-16k.wav') 

torchaudio.save("enhanced_dns4-16k.wav", est_sources[:, :, 0].detach().cpu(), 16000)