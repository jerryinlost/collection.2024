# https://github.com/microsoft/DeepSpeed/issues/4234

# llama-70b-example.py
# Launch with `deepspeed llama-70b-example.py`

import torch
import deepspeed
import os
import time
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
model_name = "meta-llama/Llama-2-70b-hf"
hf_token = "<your hf token>"


def run_zero_inference():
    ds_config = {
        "fp16": {"enabled": True},
        "bf16": {"enabled": False},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
            },
        },
        "train_micro_batch_size_per_gpu": 1,
    }
    # Share the DeepSpeed config with HuggingFace so we can properly load the
    # large model with zero stage 3
    hfdsc = HfDeepSpeedConfig(ds_config)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, torch_dtype=torch.float16
    )

    # Initialize DeepSpeed
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    # Run inference
    start_time = time.time()
    inputs = tokenizer.encode("DeepSpeed is", return_tensors="pt").to(
        f"cuda:{local_rank}"
    )
    outputs = model.generate(inputs, max_new_tokens=20)
    output_str = tokenizer.decode(outputs[0])
    end_time = time.time()
    print("ZeRO-inference time:", end_time - start_time)


def run_deepspeed_inference():
    # Load the model on meta tensors
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    with deepspeed.OnDevice(dtype=torch.float16, device="meta", enabled=True):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    # Define the checkpoint dict. You may need to convert *.safetensors to
    # *.bin for this work. Make sure you get all the *.bin and *.pt files in
    # the checkpoint_files list.
    checkpoint_dir = "~/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf/snapshots/cc8aa03a000ff08b4d5c5b39673321a2a396c396"
    checkpoint_files = [
        os.path.join(checkpoint_dir, f"model-{i:05d}-of-000015.bin")
        for i in range(1, 16)
    ]
    checkpoint_dict = {
        "type": "DS_MODEL",
        "checkpoints": checkpoint_files,
        "version": 1.0,
    }

    # Initialize DeepSpeed
    model = deepspeed.init_inference(
        model,
        replace_with_kernel_inject=False,
        mp_size=world_size,
        dtype=torch.float16,
        checkpoint=checkpoint_dict,
    )

    # Run inference
    start_time = time.time()
    inputs = tokenizer.encode("DeepSpeed is", return_tensors="pt").to(
        f"cuda:{local_rank}"
    )
    outputs = model.generate(inputs, max_new_tokens=20)
    output_str = tokenizer.decode(outputs[0])
    end_time = time.time()
    print("DeepSpeed-inference time:", end_time - start_time)


if __name__ == "__main__":
    run_zero_inference()
    run_deepspeed_inference()