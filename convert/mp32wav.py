from pydub import AudioSegment
sound = AudioSegment.from_mp3("1.mp3")
sound.export("1.wav", format="wav")