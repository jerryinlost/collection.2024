import librosa
filename='babble_15dB.opus'
y, s = librosa.load(filename, sr=16000) # Downsample 44.1kHz to 16kHz
import soundfile as sf
sf.write(f'babble_15dB_16khz.wav', y, 16000, 'PCM_16')

## The function librosa.output was removed in librosa version 0.8.0. 
# import librosa

# audio_file = "Original.wav" #48KHz

# #SAME PLAYBACK SPEED
# x, sr = librosa.load(audio_file, sr=44100)
# librosa.output.write_wav("Test1.wav", x, sr=22050, norm=False)

# #SAME PLAYBACK SPEED
# x, sr = librosa.load(audio_file, sr=48000)
# y = librosa.resample(x, 48000, 44100)
# librosa.output.write_wav("Test3.wav", y, sr=44100, norm=False)

# #SLOW PLAYBACK SPEED
# x, sr = librosa.load(audio_file, sr=48000)
# librosa.output.write_wav("Test2.wav", x, sr=44100, norm=False)

# libroa when saving the output changes the datatype to 32bit float by default. 
# Hence, to save the array, use soundwrite and specify the datatype while saving the audio