import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import json

class AudioProcessor:
    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.start_frame = None
        self.end_frame = None
        self.figure, self.ax = plt.subplots()
        self.rect = None

    def load_audio(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.wav', '.mp3', '.flac', '.ogg']:
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
        elif file_ext in ['.mp4', '.avi', '.mov']:
            clip = VideoFileClip(file_path)
            audio = clip.audio
            audio_file = 'temp_audio.wav'
            audio.write_audiofile(audio_file, codec='pcm_s16le')
            self.audio_data, self.sample_rate = librosa.load(audio_file, sr=None)
            os.remove(audio_file)
        else:
            raise ValueError("Unsupported file format")
        self.display_waveform()

    def display_waveform(self):
        self.ax.clear()
        self.ax.plot(self.audio_data)
        self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        plt.show()

    def on_click(self, event):
        if event.inaxes:
            self.start_frame = int(event.xdata)
            self.end_frame = None
            print(f"Start Frame: {self.start_frame}")

    def on_mouse_move(self, event):
        if event.inaxes and self.start_frame is not None:
            if self.rect:
                self.rect.remove()
            self.end_frame = int(event.xdata)
            self.rect = self.ax.axvspan(self.start_frame, self.end_frame, color='red', alpha=0.5)
            self.figure.canvas.draw()

    def on_release(self, event):
        if event.inaxes and self.start_frame is not None:
            self.end_frame = int(event.xdata)
            print(f"End Frame: {self.end_frame}")
            self.play_selected_audio()

    def play_selected_audio(self):
        if self.start_frame is not None and self.end_frame is not None:
            selected_audio = self.audio_data[self.start_frame:self.end_frame]
            sf.write('temp_clip.wav', selected_audio, self.sample_rate)
            os.system('play temp_clip.wav')  # Requires `sox` installed on the system

    def save_clip_and_transcription(self, transcription, output_dir):
        if self.start_frame is None or self.end_frame is None:
            print("Please select a valid audio segment first.")
            return

        os.makedirs(output_dir, exist_ok=True)
        clip_filename = f'clip_{self.start_frame}_{self.end_frame}.wav'
        clip_path = os.path.join(output_dir, clip_filename)
        sf.write(clip_path, self.audio_data[self.start_frame:self.end_frame], self.sample_rate)

        config = {
            'file': clip_filename,
            'transcription': transcription,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame
        }

        config_path = os.path.join(output_dir, 'dataset_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                dataset_config = json.load(f)
        else:
            dataset_config = []

        dataset_config.append(config)

        with open(config_path, 'w') as f:
            json.dump(dataset_config, f, indent=4)

        print(f"Clip and transcription saved to {output_dir}")

def main():
    audio_processor = AudioProcessor()
    audio_file = input("Enter the path to the audio or video file: ")
    audio_processor.load_audio(audio_file)
    
    while True:
        transcription = input("Enter the transcription for the selected audio segment: ")
        output_dir = input("Enter the directory to save the clip and transcription: ")
        audio_processor.save_clip_and_transcription(transcription, output_dir)
        
        cont = input("Do you want to process another segment? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()