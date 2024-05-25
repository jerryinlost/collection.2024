
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#"meta-llama/Meta-Llama-3-70B-Instruct"
# "garage-bAInd/Platypus2-70B-instruct"

import transformers
import torch

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

pipeline("Hey how are you doing today?")