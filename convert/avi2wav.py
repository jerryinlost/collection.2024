import subprocess

command = "ffmpeg -i input.mp4 -ab 160k -ac 2 -ar 44100 -vn output.wav"

subprocess.call(command, shell=True)

import os
os.system('ffmpeg -i input.3gp output.wav')