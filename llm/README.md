## [Download Models](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)


```sh
# default cache dir: C:\Users\xxx\.cache\huggingface\hub\models--meta-llama--Meta-Llama-3-8B-Instruct
# huggingface api token(read-only): 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

pip install -U "huggingface_hub[cli]"

huggingface-cli login
huggingface-cli download adept/fuyu-8b --cache-dir ./path/to/cache
huggingface-cli scan-cache
huggingface-cli env

# customize the default cache dir path
# python:
import os
os.environ['HF_HOME'] = '/blabla/cache/'
# bash:
# export HF_HOME=/blabla/cache/
```

## Inference

```sh
#ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`

pip install accelerate
```

### Report

```sh
AssertionError: Torch not compiled with CUDA enabled
```
- transformers: inference time is too long
- deepseed: