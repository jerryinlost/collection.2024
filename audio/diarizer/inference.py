# https://github.com/huggingface/diarizers?tab=readme-ov-file#inference-with-pyannote
from diarizers import SegmentationModel
from pyannote.audio import Pipeline
from datasets import load_dataset
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# load the pre-trained pyannote pipeline
pipeline = Pipeline.from_pretrained("./models/pyannote/speaker-diarization-3.1")
pipeline.to(device)

# replace the segmentation model with your fine-tuned one
model = SegmentationModel().from_pretrained("./models/diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn")
model = model.to_pyannote_model()
pipeline._segmentation.model = model.to(device)

# load dataset example
dataset = load_dataset("diarizers-community/callhome", "jpn", split="data")
sample = dataset[0]["audio"]

# pre-process inputs
sample["waveform"] = torch.from_numpy(sample.pop("array")[None, :]).to(device, dtype=model.dtype)
sample["sample_rate"] = sample.pop("sampling_rate")

# perform inference
diarization = pipeline(sample)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)