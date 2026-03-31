# Semi-dynamic soft context compression

This directory contains the implementation of our paper [Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio ](https://arxiv.org/abs/2603.25926). 
The model compresses long contexts into a fixed or variable number of latent tokens and injects them into the decoder at placeholder token positions.

## Files

| Path                                       | Description |
|--------------------------------------------|-------------|
| **`modeling_ctxcomp.py`**                  | Model definitions: `CtxCompModel` (fixed or multi-ratio compression) and `CtxCompSemiDynamicModel` (predicts compression ratio per sample). See docstrings for parameters (e.g. `comp_ratio_or_len`, `feature_extract_method`, `encoder` / `decoder`). |
| **`SFT/sft_ctxcomp_static.py`**            | SFT script for **static** single compression ratio/length. Trains `CtxCompModel` with one `comp_ratio_or_len`. |
| **`SFT/sft_ctxcomp_static_multiratio.py`** | SFT script for **multi-ratio** static model. Builds dataset with max ratio; collator randomly picks a ratio per batch and sets `comp_ratio_or_len_override`. |
| **`SFT/sft_ctxcomp_semi_dynamic.py`**      | SFT script for **semi-dynamic** model. Uses `CtxCompSemiDynamicModel` with single placeholder per sample and `compress_len_labels` (log2(context_len/summary_len)). |
| **`Eval/eval_ctxcomp.py`**                 | Evaluation for **CtxCompModel**: loads checkpoint, runs generation with optional `comp_ratio_or_len_override`, computes accuracy and compression ratio. Reads datasets from `Eval/eval_data/` by default. |
| **`Eval/eval_ctxcomp_semi_dynamic.py`**    | Evaluation for **CtxCompSemiDynamicModel**: one placeholder per sample, sweeps over `compress_ratio_scale`, records accuracy and (optional) dynamic compression stats. Uses `Eval/eval_data/` for datasets. |
| **`Eval/eval_data/`**                      | Default directory for evaluation datasets (e.g. `hotpot_qa/`, `squad_v2/`, `NQ_short/`, `adversarialQA/`). Place parquet files in the expected subpaths or adjust `DATASETS` in the eval scripts. |

## Model (modeling_ctxcomp.py)

- **CtxCompModel**: encoder → feature extraction (by `feature_extract_method`) → MLP projector → decoder. Placeholder token positions in the decoder input are replaced by projected compression features.
- **CtxCompSemiDynamicModel**: extends CtxCompModel with a compression-ratio head; can use a single placeholder per sample and predict M per sample (with optional `compress_ratio_scale` at inference).

Supported `feature_extract_method`: `mean_pooling`, `mean_pooling_causal`, `last_tokens`, `same_memory_tokens`, `different_memory_tokens`. 

Config main keys: `base_encoder_model_path`, `base_decoder_model_path`, `comp_ratio_or_len`, `feature_extract_method`, `placeholder_token_id` (and optional `encoder_training` / `decoder_training`).

## Data
* For evaluation, Dataset paths default to `Eval/eval_data/<dataset_name>/...`; set `DATASETS` in each script to your parquet paths.
* For training, the datasets can be downloaded from [HuggingFace](https://huggingface.co/datasets/yuyijiong/context_qa_sum_qwen3_synthetic). 
And you can set which subsets to use by modifing the ``data_config`` in the SFT scripts.

## Model weights
* Our fine-tuned LoRA model weights are available on [HuggingFace](https://huggingface.co/yuyijiong/qwen3-semi-dynamic-soft-context-compress).

---

## Quick start: load and run inference

The checkpoint directory must contain `config.json`, `encoder/`, `decoder/`, and `projector.pth` (saved by the SFT scripts). Two usage modes are supported: **fixed-ratio** (you choose compression length/ratio per call) and **semi-dynamic** (model predicts it; you can shift with `compress_ratio_scale`).

### Shared setup (path + tokenizers)

```python
import os
import sys
import torch
import json
from transformers import AutoTokenizer

# If you run from another directory, add the folder that contains modeling_ctxcomp.py
sys.path.insert(0, "/path/to/semi_dynamic_soft_context_compress")
from modeling_ctxcomp import CtxCompModel, CtxCompSemiDynamicModel

checkpoint_dir = "/path/to/your/checkpoint"  # e.g. qwen3-semi-dynamic-soft-context-compress/static/ctxcomp-encoder=qwen3-embedding-0.6b-lora-decoder=qwen3-0.6b-lora-contextlen=1300to128-ratio=0.03125

with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
    config = json.load(f)
encoder_path = config.get("base_encoder_model_path") or config.get("base_embed_model_path")
decoder_path = config.get("base_decoder_model_path") or config.get("base_gen_model_path")
placeholder_token_id = config["placeholder_token_id"]

tokenizer_encoder = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True, padding_side="left")
tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)
placeholder_token = tokenizer_decoder.convert_ids_to_tokens(placeholder_token_id)

context_text = "Your long context text here..."
question = "What is the main idea of the context?"
```

### Option A: Fixed-ratio (CtxCompModel)

Load a static or multi-ratio checkpoint; **you** set the compression length/ratio for this call via `comp_ratio_or_len_override`. The number of placeholders in the prompt must equal the resulting M (e.g. for ratio 0.25 with mean_pooling, M = ceil(context_len / 4)).

```python
model = CtxCompModel.from_pretrained(
    checkpoint_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

# Example: use ratio 0.25 for this call (yields M = ceil(context_len/4) for mean_pooling)
comp_ratio_or_len_override = 0.25  # or an int, e.g. 32, for last_tokens / memory_tokens
num_placeholders = 32  # must match M for this context and comp_ratio_or_len_override

prompt_text = f"Context: {placeholder_token * num_placeholders}\n\nQuestion: {question}"
context_encoded = tokenizer_encoder(context_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
context_input_ids = context_encoded["input_ids"].to(model.decoder.device)
context_attention_mask = context_encoded["attention_mask"].to(model.decoder.device)

messages = [{"role": "user", "content": prompt_text}]
prompt_ids = tokenizer_decoder.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
)
input_ids = prompt_ids.to(model.decoder.device)
attention_mask = torch.ones_like(input_ids, device=input_ids.device)

with torch.no_grad():
    out = model.generate(
        context_input_ids=context_input_ids,
        input_ids=input_ids,
        context_attention_mask=context_attention_mask,
        attention_mask=attention_mask,
        comp_ratio_or_len_override=comp_ratio_or_len_override,  # fixed ratio for this call
        max_new_tokens=256,
        do_sample=False,
    )

generated_ids = out[0][input_ids.shape[1]:]
answer = tokenizer_decoder.decode(generated_ids, skip_special_tokens=True)
print(answer)
```

### Option B: Semi-dynamic (CtxCompSemiDynamicModel)

Load a semi-dynamic checkpoint; the **model** predicts how many compressed tokens to use. Use **exactly one** placeholder in the prompt. You can shift the predicted ratio at inference with `compress_ratio_scale` (e.g. `0.5` = more compression, `-0.5` = less).

```python
model = CtxCompSemiDynamicModel.from_pretrained(
    checkpoint_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

# One placeholder; model predicts M and expands internally
prompt_text = f"Context: {placeholder_token}\n\nQuestion: {question}"
context_encoded = tokenizer_encoder(context_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
context_input_ids = context_encoded["input_ids"].to(model.decoder.device)
context_attention_mask = context_encoded["attention_mask"].to(model.decoder.device)

messages = [{"role": "user", "content": prompt_text}]
prompt_ids = tokenizer_decoder.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
)
input_ids = prompt_ids.to(model.decoder.device)
attention_mask = torch.ones_like(input_ids, device=input_ids.device)

# compress_ratio_scale: shift predicted ratio (e.g. 0.5 = more compression, -0.5 = less)
compress_ratio_scale = 0.5

with torch.no_grad():
    out = model.generate(
        context_input_ids=context_input_ids,
        input_ids=input_ids,
        context_attention_mask=context_attention_mask,
        attention_mask=attention_mask,
        compress_ratio_scale=compress_ratio_scale,
        max_new_tokens=256,
        do_sample=False,
    )

generated_ids = out[0][input_ids.shape[1]:]
answer = tokenizer_decoder.decode(generated_ids, skip_special_tokens=True)
print(answer)
```
