"""
Evaluation script for CtxCompSemiDynamicModel.

Single placeholder per sample; model predicts compression ratio. Evaluates over multiple
compress_ratio_scale values. Uses eval_data/ for datasets.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import json
import pathlib
import pandas as pd
from transformers import AutoTokenizer

import eval_ctxcomp as _eval

from modeling_ctxcomp import CtxCompSemiDynamicModel

acc_by_context_len = _eval.acc_by_context_len
compression_ratio_by_context_len = _eval.compression_ratio_by_context_len
aggregate_multi_dataset_results = _eval.aggregate_multi_dataset_results
prepare_eval_dataframe = _eval.prepare_eval_dataframe
stack_padded_context_input_ids = _eval.stack_padded_context_input_ids
_normalize_context_input_ids_row = _eval._normalize_context_input_ids_row
run_batch_generation = _eval.run_batch_generation
evaluate_and_append_result = _eval.evaluate_and_append_result

COMPRESS_RATIO_SCALES = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]


def aggregate_model_results_dynamic(records, identifier_keys=None, context_len_ranges=None):
    if identifier_keys is None:
        identifier_keys = ["bridge_model_path", "compress_ratio_scale"]
    return aggregate_multi_dataset_results(records, identifier_keys=identifier_keys, context_len_ranges=context_len_ranges)


def batch_generate_worker_semi_dynamic(args):
    gpu_id, df_subset, original_indices, config = args
    import pandas as pd
    from tqdm import tqdm

    if isinstance(df_subset, list):
        df_subset = pd.DataFrame(df_subset)

    bridge_model_path = config["bridge_model_path"]
    batch_size = config["batch_size"]
    compress_ratio_scale = config["compress_ratio_scale"]

    with open(os.path.join(bridge_model_path, "config.json"), "r") as f:
        model_config = json.load(f)

    model = CtxCompSemiDynamicModel.from_pretrained(
        bridge_model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": gpu_id},
    ).eval()

    embedding_path = model_config.get("base_encoder_model_path") or model_config.get("base_embed_model_path")
    generation_path = model_config.get("base_decoder_model_path") or model_config.get("base_gen_model_path")
    tokenizer_decoder = AutoTokenizer.from_pretrained(generation_path, trust_remote_code=True)
    tokenizer_decoder.padding_side = "left"
    tokenizer_encoder = AutoTokenizer.from_pretrained(embedding_path, trust_remote_code=True, padding_side="left")
    device = model.decoder.device

    df_subset = df_subset.reset_index(drop=True)
    results = []

    for batch_start in tqdm(range(0, len(df_subset), batch_size), desc=f"GPU {gpu_id}"):
        batch_end = min(batch_start + batch_size, len(df_subset))
        batch_df = df_subset.iloc[batch_start:batch_end]
        batch_templated = batch_df["templated_prompt"].tolist()
        if "context_input_ids" not in batch_df.columns:
            raise ValueError(
                "Semi-dynamic eval requires context_input_ids from prepare_eval_dataframe; re-run data prep or use force_recompute."
            )
        rows = [_normalize_context_input_ids_row(x) for x in batch_df["context_input_ids"]]
        pad_id = tokenizer_encoder.pad_token_id or tokenizer_encoder.eos_token_id
        context_input_ids, context_attention_mask = stack_padded_context_input_ids(rows, pad_id, device)
        input_encoded = tokenizer_decoder(batch_templated, return_tensors="pt", padding=True)
        input_ids = input_encoded["input_ids"].to(device)
        attention_mask = input_encoded["attention_mask"].to(device)
        eos_token_id = tokenizer_decoder.eos_token_id
        pad_token_id = tokenizer_decoder.pad_token_id or tokenizer_decoder.eos_token_id
        gen_kw = dict(
            context_input_ids=context_input_ids,
            input_ids=input_ids,
            context_attention_mask=context_attention_mask,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=512,
            compress_ratio_scale=compress_ratio_scale,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        generation_outputs = model.generate(**gen_kw)
        num_valid_tokens_batch = getattr(
            generation_outputs,
            "num_valid_tokens",
            getattr(generation_outputs, "__dict__", {}).get("num_valid_tokens"),
        )
        input_lengths = attention_mask.sum(dim=1)
        for i, output in enumerate(generation_outputs):
            generated_text = tokenizer_decoder.decode(output, skip_special_tokens=True)
            result_dict = {"index": original_indices[batch_start + i], "generated_text": generated_text}
            if num_valid_tokens_batch is not None:
                val = num_valid_tokens_batch[i]
                result_dict["num_valid_tokens"] = val.item() if hasattr(val, "item") else int(val)
            results.append(result_dict)

    return results


if __name__ == "__main__":
    context_max_length = 2048
    context_min_length = 128
    eval_acc = True
    max_sample_num_per_ds = 1000
    force_recompute = False
    batch_size = 8
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    num_processes = len(gpu_ids)
    CONTEXT_LEN_RANGES = [(1, 10000), (10000, 20000), (20000, 30000),
                          (30000, 40000)]

    bridge_model_path_list = [
        "../SFT/training_output/ctxcomp-semi-dynamic-mean_pooling-contextlen=1300to64-ratios=0.5_0.03125-enc=lora-dec=lora/checkpoint-100000"
,
    ]

    EVAL_DATA_DIR = pathlib.Path(__file__).resolve().parent / "eval_data"
    DATASETS = {
        "hotpotqa": str(EVAL_DATA_DIR / "hotpotqa_validation_cqa.parquet"),
        "squad": str(EVAL_DATA_DIR / "squad_validation_cqa.parquet"),
        "NQ": str(EVAL_DATA_DIR / "NQ_validation_cqa.parquet"),
        "adversarialQA": str(EVAL_DATA_DIR / "adverserialqa_validation_cqa.parquet")}


    dataset_names_sorted = sorted(DATASETS.keys())
    multi_dataset = len(DATASETS) > 1
    summary_jsonl_path = (
        f"eval_aggregate_ctxcomp_dynamic_{'_'.join(dataset_names_sorted)}_contextlen={context_min_length}to{context_max_length}.jsonl"
        if multi_dataset
        else None
    )

    for model_idx, bridge_model_path in enumerate(bridge_model_path_list):
        print(f"\n{'='*80}\nModel {model_idx+1}/{len(bridge_model_path_list)}: {bridge_model_path}")
        with open(os.path.join(bridge_model_path, "config.json"), "r") as f:
            model_config = json.load(f)
        comp_ratio_or_len = model_config.get("comp_ratio_or_len", model_config.get("num_doc_tokens"))
        if not isinstance(comp_ratio_or_len, (list, tuple)) or len(comp_ratio_or_len) == 0:
            print("Skip: CtxCompSemiDynamicModel requires non-empty comp_ratio_or_len list in config.")
            continue
        generation_path = model_config.get("base_decoder_model_path") or model_config.get("base_gen_model_path")
        embedding_path = model_config.get("base_encoder_model_path") or model_config.get("base_embed_model_path")
        placeholder_token_id = model_config["placeholder_token_id"]
        feature_extract_method = model_config.get("feature_extract_method", model_config.get("compress_method", "mean_pooling"))
        tokenizer_decoder = AutoTokenizer.from_pretrained(generation_path, trust_remote_code=True)
        tokenizer_encoder = AutoTokenizer.from_pretrained(embedding_path, trust_remote_code=True)
        placeholder_token = tokenizer_decoder.convert_ids_to_tokens(placeholder_token_id)

        for current_scale in COMPRESS_RATIO_SCALES:
            model_records = []
            print(f"\n--- Evaluating compress_ratio_scale: {current_scale} ---")
            for dataset_name, data_path in DATASETS.items():
                if not os.path.isfile(data_path):
                    print(f"Skip (file not found): {data_path}")
                    continue
                print(f"\n>>> Dataset: {dataset_name} ({data_path})")
                results_jsonl_path = (
                    None
                    if multi_dataset
                    else f"eval_ctxcomp_dynamic_{dataset_name}_contextlen={context_min_length}to{context_max_length}.jsonl"
                )
                save_dir = f"eval_results/{dataset_name}"
                base_name = f"{pathlib.Path(bridge_model_path).name}"
                file_name = f"{base_name}-scale_{current_scale}.parquet"
                save_path = os.path.join(save_dir, file_name)
                df_exists = os.path.isfile(save_path)

                if df_exists and not force_recompute:
                    print(f"Loading existing results: {save_path}")
                    df_filtered = pd.read_parquet(save_path).reset_index(drop=True)
                else:
                    df = pd.read_parquet(data_path)
                    df_filtered = prepare_eval_dataframe(
                        df,
                        tokenizer_decoder=tokenizer_decoder,
                        context_min_length=context_min_length,
                        context_max_length=context_max_length,
                        max_sample_num_per_ds=max_sample_num_per_ds,
                        tokenizer_encoder=tokenizer_encoder,
                        is_bridge=True,
                        placeholder_token=placeholder_token,
                        feature_extract_method=feature_extract_method,
                        comp_ratio_or_len=None,
                        single_placeholder=True,
                        add_eos_token_to_context=True,
                    )
                    config = {
                        "bridge_model_path": bridge_model_path,
                        "compress_ratio_scale": current_scale,
                        "batch_size": batch_size,
                    }
                    df_filtered = run_batch_generation(
                        df_filtered, config, gpu_ids, num_processes, batch_generate_worker_semi_dynamic
                    )

                if eval_acc:
                    result_record = evaluate_and_append_result(
                        df_filtered,
                        dataset_name,
                        bridge_model_path,
                        results_jsonl_path,
                        is_bridge=True,
                        comp_ratio_or_len=None,
                        feature_extract_method=feature_extract_method,
                        is_dynamic=True,
                        comp_ratio_or_len_for_dynamic=comp_ratio_or_len,
                        extra_result_record={"compress_ratio_scale": current_scale},
                        context_len_ranges=CONTEXT_LEN_RANGES,
                    )
                    model_records.append(result_record.copy())

                if (not df_exists) or force_recompute:
                    torch.cuda.empty_cache()
                    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                    df_filtered.to_parquet(save_path)

            if model_records:
                if multi_dataset and summary_jsonl_path:
                    summary = aggregate_model_results_dynamic(model_records, context_len_ranges=CONTEXT_LEN_RANGES)
                    pathlib.Path(summary_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(summary_jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                    print(f"\nAggregated result appended to {summary_jsonl_path} (scale={current_scale})")
#nohup python eval_ctxcomp_semi_dynamic.py > eval_comp_semi_dynamic.log 2>&1 &