# python model-qa.py -m /*{YourModelPath}*/onnx/cpu_and_mobile/phi-3-mini-4k-instruct-int4-cpu -k 40 -p 0.95 -t 0.8 -r 1.0

"""
F:\Models\llm\llama3>huggingface-cli download microsoft/Phi-3-mini-128k-instruct-onnx --include "*.onnx"
Fetching 5 files:   0%|                                                                          | 0/5 [00:00<?, ?it/s]Downloading 'cpu_and_mobile/cpu-int4-rtn-block-32/phi3-mini-128k-instruct-cpu-int4-rtn-block-32.onnx' to 'C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\cbcb209993e7508380321a5b57f33c24c5c20ae3f9d22f6cc6b51c5f4bdab79a.incomplete'
Downloading 'directml/directml-int4-awq-block-128/model.onnx' to 'C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\eac80c41ab030386a8074bb04a43f05a1572b4efaf7fca3a3d3bafd1b4aaeeb7.incomplete'
Downloading 'cuda/cuda-int4-rtn-block-32/phi3-mini-128k-instruct-cuda-int4-rtn-block-32.onnx' to 'C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\89ecae6d6e74c4d15bdf6dad8fa11c4f56f79328cd1623d336bae52693538e75.incomplete'
Downloading 'cuda/cuda-fp16/phi3-mini-128k-instruct-cuda-fp16.onnx' to 'C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\543dbe021574541134fdd975a9c22b80c38ca3d8ec7138db937fdd93b0914a99.incomplete'
Downloading 'cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx' to 'C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\ab01ff406f32f83d3954b53976ccc70d070b4186cbd7c46da7b4f6483c18ab9a.incomplete'
(…)28k-instruct-cuda-int4-rtn-block-32.onnx: 100%|████████████████████████████████| 26.2M/26.2M [00:12<00:00, 2.06MB/s]
Download complete. Moving file to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\89ecae6d6e74c4d15bdf6dad8fa11c4f56f79328cd1623d336bae52693538e75                     | 0.00/52.2M [00:00<?, ?B/s]
C:\Users\xxx\AppData\Local\Programs\Python\Python310\lib\site-packages\huggingface_hub\file_download.py:157: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to see activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Symlink not supported. Moving file from C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\89ecae6d6e74c4d15bdf6dad8fa11c4f56f79328cd1623d336bae52693538e75 to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\snapshots\791e509f326110e83437e537c2c4182815a6819a\cuda\cuda-int4-rtn-block-32\phi3-mini-128k-instruct-cuda-int4-rtn-block-32.onnx
phi3-mini-128k-instruct-cuda-fp16.onnx: 100%|██████████████████████████████████████| 26.1M/26.1M [00:30<00:00, 848kB/s]
Download complete. Moving file to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\543dbe021574541134fdd975a9c22b80c38ca3d8ec7138db937fdd93b0914a99             | 10.5M/52.2M [00:22<01:28, 472kB/s]
Symlink not supported. Moving file from C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\543dbe021574541134fdd975a9c22b80c38ca3d8ec7138db937fdd93b0914a99 to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\snapshots\791e509f326110e83437e537c2c4182815a6819a\cuda\cuda-fp16\phi3-mini-128k-instruct-cuda-fp16.onnx
model.onnx: 100%|██████████████████████████████████████████████████████████████████| 32.6M/32.6M [00:43<00:00, 745kB/s]
Download complete. Moving file to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\eac80c41ab030386a8074bb04a43f05a1572b4efaf7fca3a3d3bafd1b4aaeeb7             | 31.5M/52.2M [00:39<00:23, 900kB/s]
Symlink not supported. Moving file from C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\eac80c41ab030386a8074bb04a43f05a1572b4efaf7fca3a3d3bafd1b4aaeeb7 to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\snapshots\791e509f326110e83437e537c2c4182815a6819a\directml\directml-int4-awq-block-128\model.onnx
(…)128k-instruct-cpu-int4-rtn-block-32.onnx: 100%|████████████████████████████████| 52.2M/52.2M [00:51<00:00, 1.01MB/s]
Download complete. Moving file to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\cbcb209993e7508380321a5b57f33c24c5c20ae3f9d22f6cc6b51c5f4bdab79a████████████| 52.2M/52.2M [00:51<00:00, 1.28MB/s]
Symlink not supported. Moving file from C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\cbcb209993e7508380321a5b57f33c24c5c20ae3f9d22f6cc6b51c5f4bdab79a to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\snapshots\791e509f326110e83437e537c2c4182815a6819a\cpu_and_mobile\cpu-int4-rtn-block-32\phi3-mini-128k-instruct-cpu-int4-rtn-block-32.onnx
(…)t-cpu-int4-rtn-block-32-acc-level-4.onnx: 100%|█████████████████████████████████| 52.2M/52.2M [00:52<00:00, 989kB/s]
Download complete. Moving file to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\ab01ff406f32f83d3954b53976ccc70d070b4186cbd7c46da7b4f6483c18ab9a
Symlink not supported. Moving file from C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\blobs\ab01ff406f32f83d3954b53976ccc70d070b4186cbd7c46da7b4f6483c18ab9a to C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\snapshots\791e509f326110e83437e537c2c4182815a6819a\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4\phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx
Fetching 5 files: 100%|██████████████████████████████████████████████████████████████████| 5/5 [00:55<00:00, 11.09s/it]
C:\Users\xxx\.cache\huggingface\hub\models--microsoft--Phi-3-mini-128k-instruct-onnx\snapshots\791e509f326110e83437e537c2c4182815a6819a
"""
