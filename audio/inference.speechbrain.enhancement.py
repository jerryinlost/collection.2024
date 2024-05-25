from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/rescuespeech_sepformer", savedir='pretrained_models/rescuespeech_sepformer')

# for custom file, change path
est_sources = model.separate_file(path='speechbrain/rescuespeech_sepformer/example_rescuespeech16k.wav') 

torchaudio.save("enhanced_rescuespeech16k.wav", est_sources[:, :, 0].detach().cpu(), 16000)