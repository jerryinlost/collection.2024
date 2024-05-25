```sh
# Changing sample format and bit depth on audio files with ffmpeg
# exported all the songs in 48 kHz and 24 bit. Now I need it in 44.1 and 16 bit 
ffmpeg -i input.wav -ar 44100 output.wav
ffmpeg -i input.wav -c:a pcm_s16le -ar 44100 output.wav
ffmpeg -i input.wav -sample_fmt s16 -ar 44100 output.wav
ffmpeg -i input.wav -af "aformat=sample_fmts=s16:sample_rates=44100" output.wav

# See a list of encoders
ffmpeg -encoders 

# FFmpeg's FLAC encoder supports sample bit depths of 16 and 24 bits, the latter padded to 32-bit. So for 24-bit, you will have to use a filter in-between.
ffmpeg -i in.wav -af aformat=s32:176000 out.flac

#The above encodes to a 176 kHz 24-bit sample, stored as 32-bits. And the command below encodes to 16-bit and 44.1 kHz.
ffmpeg -i in.wav -af aformat=s16:44100 out.flac
```