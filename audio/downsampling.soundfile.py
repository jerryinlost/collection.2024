import soundfile as sf
data, samplerate = sf.read('old.wav')
sf.write("Test4.wav", data, 16000, subtype='PCM_16')