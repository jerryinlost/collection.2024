# https://huggingface.co/speechbrain/asr-transformer-aishell
from speechbrain.inference.ASR import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-aishell", savedir="pretrained_models/asr-transformer-aishell")
asr_model.transcribe_file("speechbrain/asr-transformer-aishell/example_mandarin2.flac")

# train
# cd recipes/AISHELL-1/ASR/transformer/
# python train.py hparams/train_ASR_transformer.yaml --data_folder=your_data_folder
