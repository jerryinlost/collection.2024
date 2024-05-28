import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import simpleaudio as sa

class AudioProcessor:
    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.start_frame = None
        self.end_frame = None
        self.figure, self.ax = plt.subplots(figsize=(15, 5))  # Set larger figure size
        self.rect = None
        self.hline = self.ax.axhline(color='r', linewidth=0.8)  # Horizontal line
        self.vline = self.ax.axvline(color='r', linewidth=0.8)  # Vertical line
        self.downsampled_data = None
        self.play_obj = None

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
        self.downsample_audio()
        self.display_waveform()

    def downsample_audio(self, factor=10):
        self.downsampled_data = self.audio_data[::factor]  # Downsample by taking every nth sample
        self.downsampled_rate = self.sample_rate // factor

    def display_waveform(self):
        self.ax.clear()
        self.ax.plot(self.downsampled_data)
        self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.figure.canvas.draw()

    def on_click(self, event):
        if event.inaxes:
            self.start_frame = int(event.xdata * 10)  # Adjust for downsampling
            self.end_frame = None
            print(f"Start Frame: {self.start_frame}")

    def on_mouse_move(self, event):
        if event.inaxes and self.start_frame is not None:
            if self.rect:
                self.rect.remove()
            self.end_frame = int(event.xdata * 10)  # Adjust for downsampling
            self.rect = self.ax.axvspan(self.start_frame // 10, self.end_frame // 10, color='red', alpha=0.5)
            self.hline.set_ydata(event.ydata)
            self.vline.set_xdata(event.xdata)
            self.figure.canvas.draw()

    def on_release(self, event):
        if event.inaxes and self.start_frame is not None:
            self.end_frame = int(event.xdata * 10)  # Adjust for downsampling
            print(f"End Frame: {self.end_frame}")
            self.play_selected_audio()

    def play_selected_audio(self):
        if self.start_frame is not None and self.end_frame is not None:
            selected_audio = self.audio_data[self.start_frame:self.end_frame]
            sf.write('temp_clip.wav', selected_audio, self.sample_rate)
            if self.play_obj:
                self.play_obj.stop()
            wave_obj = sa.WaveObject.from_wave_file('temp_clip.wav')
            self.play_obj = wave_obj.play()

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

class SpeechRecognitionDatasetApp:
    def __init__(self, root):
        self.root = root
        self.audio_processor = AudioProcessor()

        self.root.title("Speech Recognition Dataset Tool")
        self.root.geometry("800x600")

        self.load_button = tk.Button(root, text="Load Audio/Video File", command=self.load_file)
        self.load_button.pack(pady=10)

        self.transcription_label = tk.Label(root, text="Transcription:")
        self.transcription_label.pack(pady=5)
        self.transcription_entry = tk.Entry(root, width=50)
        self.transcription_entry.pack(pady=5)

        self.play_button = tk.Button(root, text="Play Selected Clip", command=self.play_clip)
        self.play_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Clip and Transcription", command=self.save_clip_and_transcription)
        self.save_button.pack(pady=10)

        self.output_dir_label = tk.Label(root, text="Output Directory:")
        self.output_dir_label.pack(pady=5)
        self.output_dir_entry = tk.Entry(root, width=50)
        self.output_dir_entry.pack(pady=5)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.audio_processor.figure, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(title="Select Audio or Video File")
        if file_path:
            try:
                self.audio_processor.load_audio(file_path)
                self.canvas.draw()
            except ValueError as e:
                messagebox.showerror("Error", str(e))

    def play_clip(self):
        self.audio_processor.play_selected_audio()

    def save_clip_and_transcription(self):
        transcription = self.transcription_entry.get()
        output_dir = self.output_dir_entry.get()
        if transcription and output_dir:
            self.audio_processor.save_clip_and_transcription(transcription, output_dir)
        else:
            messagebox.showwarning("Warning", "Please provide both transcription and output directory")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionDatasetApp(root)
    root.mainloop()