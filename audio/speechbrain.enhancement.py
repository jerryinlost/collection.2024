from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')

# for custom file, change path
est_sources = model.separate_file(path='speechbrain/sepformer-whamr-enhancement/example_whamr.wav') 

torchaudio.save("enhanced_whamr.wav", est_sources[:, :, 0].detach().cpu(), 8000)