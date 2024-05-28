import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QLineEdit
from pydub import AudioSegment
import simpleaudio as sa

class AudioProcessor:
    def __init__(self):
        self.audio_files = []
        self.play_obj = None

    def add_audio_file(self, file_path):
        self.audio_files.append(file_path)

    def play_audio_file(self, file_path):
        audio = AudioSegment.from_file(file_path)
        audio.export('temp_play.wav', format='wav')
        wave_obj = sa.WaveObject.from_wave_file('temp_play.wav')
        if self.play_obj:
            self.play_obj.stop()
        self.play_obj = wave_obj.play()

    def change_sample_rate(self, file_path, sample_rate):
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(sample_rate)
        return audio

    def merge_audio_files(self, output_path):
        if not self.audio_files:
            return None

        combined = AudioSegment.from_file(self.audio_files[0])
        for file_path in self.audio_files[1:]:
            audio = AudioSegment.from_file(file_path)
            combined += audio

        combined.export(output_path, format="wav")
        return output_path

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.audio_processor = AudioProcessor()

        self.setWindowTitle("Audio File Merger")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.file_list)

        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Add Audio File")
        self.add_button.clicked.connect(self.add_audio_file)
        button_layout.addWidget(self.add_button)

        self.play_button = QPushButton("Play Selected File")
        self.play_button.clicked.connect(self.play_selected_file)
        button_layout.addWidget(self.play_button)

        self.sample_rate_label = QLabel("Sample Rate:")
        button_layout.addWidget(self.sample_rate_label)

        self.sample_rate_input = QLineEdit()
        self.sample_rate_input.setPlaceholderText("e.g. 44100")
        button_layout.addWidget(self.sample_rate_input)

        self.change_sample_rate_button = QPushButton("Change Sample Rate")
        self.change_sample_rate_button.clicked.connect(self.change_sample_rate)
        button_layout.addWidget(self.change_sample_rate_button)

        layout.addLayout(button_layout)

        self.merge_button = QPushButton("Merge Audio Files")
        self.merge_button.clicked.connect(self.merge_audio_files)
        layout.addWidget(self.merge_button)

        self.setLayout(layout)

    def add_audio_file(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select Audio Files", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")
        if file_paths:
            for file_path in file_paths:
                self.audio_processor.add_audio_file(file_path)
                self.file_list.addItem(file_path)

    def play_selected_file(self):
        selected_item = self.file_list.currentItem()
        if selected_item:
            file_path = selected_item.text()
            self.audio_processor.play_audio_file(file_path)

    def change_sample_rate(self):
        selected_item = self.file_list.currentItem()
        if selected_item:
            file_path = selected_item.text()
            try:
                sample_rate = int(self.sample_rate_input.text())
                new_audio = self.audio_processor.change_sample_rate(file_path, sample_rate)
                new_audio.export(file_path, format="wav")
                QMessageBox.information(self, "Success", f"Sample rate changed to {sample_rate} Hz")
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter a valid sample rate")

    def merge_audio_files(self):
        file_dialog = QFileDialog()
        output_path, _ = file_dialog.getSaveFileName(self, "Save Merged Audio File", "", "WAV Files (*.wav)")
        if output_path:
            merged_path = self.audio_processor.merge_audio_files(output_path)
            if merged_path:
                QMessageBox.information(self, "Success", f"Audio files merged and saved to {merged_path}")
            else:
                QMessageBox.warning(self, "Error", "No audio files to merge")

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()