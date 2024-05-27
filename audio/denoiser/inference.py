# C:\Users\xxx/.cache\torch\hub\checkpoints\
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

model = pretrained.dns64().cpu()

wav, sr = torchaudio.load('./samples/babble_15dB_16000.wav')#.opus')

wav = convert_audio(wav.cpu(),sr,model.sample_rate,model.chin) #.cuda()

with torch.no_grad():
    denoised = model(wav[None])[0]

# soundfile.write('after_noisy.wav',denoised.data.cpu().numpy().T,sr)
torchaudio.save(f'after_noisy_{sr}.wav',denoised,sr)