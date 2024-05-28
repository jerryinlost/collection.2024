import sys
import os
import time
import numpy as np
import librosa
import soundfile as sf
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import json
import simpleaudio as sa

class AudioProcessor:
    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.start_frame = None
        self.end_frame = None
        self.downsampled_data = None
        self.downsampling_factor = 10
        self.play_obj = None
        self.start_time = None

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

    def downsample_audio(self):
        self.downsampled_data = self.audio_data[::self.downsampling_factor]  # Downsample by taking every nth sample
        self.downsampled_rate = self.sample_rate // self.downsampling_factor

    def play_selected_audio(self):
        if self.start_frame is not None and self.end_frame is not None:
            selected_audio = self.audio_data[self.start_frame:self.end_frame]
            sf.write('temp_clip.wav', selected_audio, self.sample_rate)
            if self.play_obj:
                self.play_obj.stop()
            wave_obj = sa.WaveObject.from_wave_file('temp_clip.wav')
            self.play_obj = wave_obj.play()
            self.start_time = time.time()  # Record the start time

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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.audio_processor = AudioProcessor()

        self.setWindowTitle("Speech Recognition Dataset Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.load_button = QtWidgets.QPushButton("Load Audio/Video File")
        self.load_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_button)

        self.transcription_label = QtWidgets.QLabel("Transcription:")
        self.layout.addWidget(self.transcription_label)
        self.transcription_entry = QtWidgets.QLineEdit()
        self.layout.addWidget(self.transcription_entry)

        self.play_button = QtWidgets.QPushButton("Play Selected Clip")
        self.play_button.clicked.connect(self.play_clip)
        self.layout.addWidget(self.play_button)

        self.save_button = QtWidgets.QPushButton("Save Clip and Transcription")
        self.save_button.clicked.connect(self.save_clip_and_transcription)
        self.layout.addWidget(self.save_button)

        self.output_dir_label = QtWidgets.QLabel("Output Directory:")
        self.layout.addWidget(self.output_dir_label)
        self.output_dir_entry = QtWidgets.QLineEdit()
        self.layout.addWidget(self.output_dir_entry)

        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.plot_widget.addItem(self.region)
        self.region.sigRegionChanged.connect(self.update_region)

        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.getPlotItem().getAxis('bottom').setTickSpacing(1, 0.1)

        # Color the entire plot area green
        self.green_region = pg.LinearRegionItem([0, 0], brush=(0, 255, 0, 50))
        self.green_region.setZValue(5)
        self.plot_widget.addItem(self.green_region)

        self.vertical_line = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.plot_widget.addItem(self.vertical_line)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_vertical_line)

    def load_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Audio or Video File")
        if file_path:
            try:
                self.audio_processor.load_audio(file_path)
                self.plot_audio()
            except ValueError as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def plot_audio(self):
        self.plot_widget.clear()
        downsampled_data = self.audio_processor.downsampled_data
        time_axis = np.linspace(0, len(downsampled_data) / self.audio_processor.downsampled_rate, len(downsampled_data))
        self.plot_widget.plot(time_axis, downsampled_data, pen="b")
        self.region.setRegion([0, len(downsampled_data) / self.audio_processor.downsampled_rate])

    def update_region(self):
        min_x, max_x = self.region.getRegion()
        self.audio_processor.start_frame = int(min_x * self.audio_processor.sample_rate)  # Convert to original sample rate
        self.audio_processor.end_frame = int(max_x * self.audio_processor.sample_rate)  # Convert to original sample rate
        print(f"Start Frame: {self.audio_processor.start_frame}, End Frame: {self.audio_processor.end_frame}")

        # Update the green and red regions
        self.green_region.setRegion([0, min_x])
        self.red_region = pg.LinearRegionItem([min_x, max_x], brush=(255, 0, 0, 50))
        self.red_region.setZValue(5)
        self.plot_widget.addItem(self.red_region)

    def play_clip(self):
        self.audio_processor.play_selected_audio()

        # Calculate the interval for the timer
        interval = 1000 // self.audio_processor.sample_rate
        self.timer.start(interval)  # Update every millisecond

    def update_vertical_line(self):
        if self.audio_processor.play_obj and self.audio_processor.play_obj.is_playing():
            elapsed_time = time.time() - self.audio_processor.start_time
            self.vertical_line.setPos(elapsed_time)
        else:
            self.timer.stop()

    def save_clip_and_transcription(self):
        transcription = self.transcription_entry.text()
        output_dir = self.output_dir_entry.text()
        if transcription and output_dir:
            self.audio_processor.save_clip_and_transcription(transcription, output_dir)
            self.plot_widget.removeItem(self.red_region)
            self.green_region.setRegion([0, self.audio_processor.end_frame / self.audio_processor.sample_rate])
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please provide both transcription and output directory")

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()