"""
Evaluation script for CtxCompModel (static / multi-ratio).

Uses modeling_ctxcomp.CtxCompModel. Config: base_encoder_model_path, base_decoder_model_path,
comp_ratio_or_len, feature_extract_method, placeholder_token_id.
Evaluation data under eval_data/ by default.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import math
import multiprocessing as mp
import pandas as pd
import json
import pathlib
import re
import numpy as np
# import transformers.utils.import_utils as _tiu
# _tiu.is_causal_conv1d_available = lambda: False
# _tiu.is_flash_linear_attention_available=lambda: False
from transformers import AutoTokenizer, AutoModelForCausalLM,Qwen3ForCausalLM,Qwen3_5ForCausalLM


from modeling_ctxcomp import CtxCompModel

os.environ.setdefault("WANDB_PROJECT", "SFT_ctxcomp")
mp.set_start_method("spawn", force=True)



def normalize_for_acc(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w]", "", text).replace("_", "")
    return text


def acc_by_context_len(df, response_col="generated_text", recompute_acc=False, context_len_ranges=None):
    def _get_ref_texts(row):
        if "answers" in df.columns:
            ans = row.get("answers")
            if isinstance(ans, dict) and "text" in ans:
                t = ans["text"]
                return t if isinstance(t, list) or isinstance(t, np.ndarray) else [t]
            if isinstance(ans, (list, np.ndarray)):
                return list(ans)
            raise ValueError(f"Invalid answers: {ans}")
        if "answer" in df.columns:
            a = row.get("answer")
            if isinstance(a, str) and a:
                return [a]
            raise ValueError(f"Invalid answer: {a}")
        raise ValueError(f"No answers/answer in {df.columns}")

    def _is_correct(row):
        pred_norm = normalize_for_acc(row[response_col])
        for ref in _get_ref_texts(row):
            if normalize_for_acc(ref) in pred_norm:
                return 1
        return 0

    if "is_correct" not in df.columns or recompute_acc:
        df["is_correct"] = df.apply(_is_correct, axis=1)
    accuracy = df["is_correct"].mean()
    print(f"Overall Accuracy of this dataset: {accuracy * 100:.2f}%")
    sub_accuracy_list = []
    for start, end in (context_len_ranges or []):
        sub_df = df[(df["context_len"] >= start) & (df["context_len"] < end)]
        if len(sub_df) > 0:
            sub_acc = sub_df["is_correct"].mean()
            sub_accuracy_list.append(round(float(sub_acc) * 100, 1))
            print(f"Context Length {start}-{end}: Accuracy: {sub_acc * 100:.2f}%, Samples: {len(sub_df)}")
        else:
            sub_accuracy_list.append(None)
            print(f"Context Length {start}-{end}: No samples")
    return {"overall_accuracy": round(float(accuracy) * 100, 1), "context_len_accuracies": sub_accuracy_list, "num_samples": len(df)}


def compression_ratio_by_context_len(df, comp_ratio_or_len=None, is_dynamic=False, context_len_ranges=None):
    if "compressed_len" in df.columns:
        df_valid = df.copy()
        df_valid["compression_ratio"] = df_valid["context_len"] / df_valid["compressed_len"]
    elif "num_valid_tokens" in df.columns:
        valid = df["num_valid_tokens"] > 0
        df_valid = df.loc[valid].copy()
        df_valid["compression_ratio"] = df_valid["context_len"] / df_valid["num_valid_tokens"]
    else:
        df_valid = df.copy()
        df_valid["compression_ratio"] = df_valid["context_len"] / comp_ratio_or_len
    if len(df_valid) == 0:
        return None
    overall_ratio = df_valid["compression_ratio"].mean()
    print(f"Overall Avg Compression Ratio: {overall_ratio:.2f}")
    sub_ratio_list = []
    for start, end in (context_len_ranges or []):
        sub_df = df_valid[(df_valid["context_len"] >= start) & (df_valid["context_len"] < end)]
        if len(sub_df) > 0:
            sub_ratio_list.append(round(float(sub_df["compression_ratio"].mean()), 2))
        else:
            sub_ratio_list.append(None)
    out = {"avg_comp_ratio": round(float(overall_ratio), 2), "comp_ratio_by_ctx_len": sub_ratio_list}
    if "is_correct" in df_valid.columns:
        df_correct = df_valid[df_valid["is_correct"].astype(bool)]
        if len(df_correct) > 0:
            out["avg_comp_ratio_correct"] = round(float(df_correct["compression_ratio"].mean()), 2)
            correct_context_len_sum = int(df_correct["context_len"].sum())
            if "compressed_len" in df_correct.columns:
                correct_compressed_sum = int(df_correct["compressed_len"].sum())
            elif "num_valid_tokens" in df_correct.columns:
                correct_compressed_sum = int(df_correct["num_valid_tokens"].sum())
            else:
                correct_compressed_sum = len(df_correct) * int(comp_ratio_or_len)
            out["correct_context_len_sum"] = correct_context_len_sum
            out["correct_compressed_sum"] = correct_compressed_sum
            if correct_compressed_sum > 0:
                out["avg_comp_ratio_correct_by_tokens"] = round(correct_context_len_sum / correct_compressed_sum, 2)
                print(f"Avg Compression Ratio by tokens (correct only): {out['avg_comp_ratio_correct_by_tokens']:.2f}")
    return out


def aggregate_multi_dataset_results(records, identifier_keys, context_len_ranges=None):
    if not records:
        return None
    num_buckets = len(context_len_ranges) if context_len_ranges else 0
    total_samples = sum(r["num_samples"] for r in records)
    if total_samples == 0:
        return None
    total_correct = sum(r["overall_accuracy"] / 100.0 * r["num_samples"] for r in records)
    total_accuracy = round(total_correct / total_samples * 100, 1)
    per_dataset_accuracy = {r["dataset"]: r["overall_accuracy"] for r in records}
    context_len_accuracies = []
    for i in range(num_buckets):
        vals = [(r["context_len_accuracies"][i], r["num_samples"]) for r in records if i < len(r["context_len_accuracies"]) and r["context_len_accuracies"][i] is not None]
        if vals:
            s = sum(v * w for v, w in vals)
            w = sum(w for _, w in vals)
            context_len_accuracies.append(round(s / w, 1))
        else:
            context_len_accuracies.append(None)
    out = {k: records[0][k] for k in identifier_keys if k in records[0]}
    out.update({
        "total_accuracy": total_accuracy,
        "per_dataset_accuracy": per_dataset_accuracy,
        "context_len_accuracies": context_len_accuracies,
        "total_num_samples": total_samples,
        "per_dataset_num_samples": {r["dataset"]: r["num_samples"] for r in records},
    })
    records_with_comp = [r for r in records if r.get("avg_comp_ratio") is not None]
    if records_with_comp:
        w_total = sum(r["num_samples"] for r in records_with_comp)
        out["avg_comp_ratio"] = round(sum(r["avg_comp_ratio"] * r["num_samples"] for r in records_with_comp) / w_total, 2)
        out["comp_ratio_by_ctx_len"] = []
        for i in range(num_buckets):
            vals = [(r["comp_ratio_by_ctx_len"][i], r["num_samples"]) for r in records_with_comp if i < len(r["comp_ratio_by_ctx_len"]) and r["comp_ratio_by_ctx_len"][i] is not None]
            if vals:
                s = sum(v * w for v, w in vals)
                w = sum(w for _, w in vals)
                out["comp_ratio_by_ctx_len"].append(round(s / w, 2))
            else:
                out["comp_ratio_by_ctx_len"].append(None)
        records_with_correct = [r for r in records_with_comp if r.get("avg_comp_ratio_correct") is not None]
        if records_with_correct:
            num_correct_list = [r["overall_accuracy"] / 100.0 * r["num_samples"] for r in records_with_correct]
            w_correct_total = sum(num_correct_list)
            out["avg_comp_ratio_correct"] = round(
                sum(r["avg_comp_ratio_correct"] * w for r, w in zip(records_with_correct, num_correct_list)) / w_correct_total, 2
            ) if w_correct_total > 0 else None
        records_with_tokens = [r for r in records_with_comp if r.get("correct_context_len_sum") is not None and r.get("correct_compressed_sum") is not None]
        if records_with_tokens:
            total_ctx_correct = sum(r["correct_context_len_sum"] for r in records_with_tokens)
            total_comp_correct = sum(r["correct_compressed_sum"] for r in records_with_tokens)
            if total_comp_correct > 0:
                out["avg_comp_ratio_correct_by_tokens"] = round(total_ctx_correct / total_comp_correct, 2)
    return out


def aggregate_model_results(records, identifier_keys=None, context_len_ranges=None):
    if identifier_keys is None:
        identifier_keys = ["bridge_model_path", "comp_ratio_or_len"] if records and "comp_ratio_or_len" in records[0] else ["bridge_model_path"]
    return aggregate_multi_dataset_results(records, identifier_keys, context_len_ranges=context_len_ranges)


def _ratio_based(feature_extract_method):
    return feature_extract_method in ("mean_pooling", "mean_pooling_causal")


def _tokenize_context_with_optional_eos(text, tokenizer_encoder, add_eos_token_to_context: bool):
    encodings = tokenizer_encoder(text)
    input_ids = list(encodings["input_ids"])
    if add_eos_token_to_context:
        eos_id = tokenizer_encoder.eos_token_id
        if eos_id is None:
            raise ValueError(
                "tokenize_context: tokenizer has no eos_token_id; define eos on the tokenizer or pass add_eos_token_to_context=False"
            )
        if not input_ids or input_ids[-1] != eos_id:
            input_ids.append(eos_id)
    return input_ids


def _normalize_context_input_ids_row(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return [int(t) for t in x]


def stack_padded_context_input_ids(rows, pad_id, device):
    """Pad variable-length context token id lists to a batch tensor (left padding, content right-aligned)."""
    max_len = max(len(r) for r in rows)
    context_input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    context_attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(rows):
        L = len(ids)
        context_input_ids[i, max_len - L:] = torch.tensor(ids, dtype=torch.long, device=device)
        context_attention_mask[i, max_len - L:] = 1
    return context_input_ids, context_attention_mask


def prepare_eval_dataframe(
    df,
    *,
    tokenizer_decoder,
    context_min_length,
    context_max_length,
    max_sample_num_per_ds=None,
    tokenizer_encoder=None,
    is_bridge=False,
    placeholder_token=None,
    feature_extract_method=None,
    comp_ratio_or_len=None,
    single_placeholder=False,
    add_eos_token_to_context: bool = True,
):
    df = df.reset_index(drop=True)
    if is_bridge:
        if tokenizer_encoder is None:
            raise ValueError("is_bridge=True requires tokenizer_encoder")
        df["context_input_ids"] = df["context"].apply(
            lambda x: _tokenize_context_with_optional_eos(x, tokenizer_encoder, add_eos_token_to_context)
        )
        df["context_len"] = df["context_input_ids"].apply(len)
        if not single_placeholder:
            if _ratio_based(feature_extract_method or "") and isinstance(comp_ratio_or_len, (float, int)) and 0 < comp_ratio_or_len <= 1:
                pool_size = max(1, round(1.0 / float(comp_ratio_or_len)))
                df["compressed_len"] = df["context_len"].apply(lambda cl: max(1, (cl + pool_size - 1) // pool_size))
            else:
                n_placeholders = int(comp_ratio_or_len) if comp_ratio_or_len is not None else 64
                df["compressed_len"] = n_placeholders
        if single_placeholder:
            df["padded_question"] = df.apply(lambda row: f"Context: {placeholder_token}\nQuestion: {row['question']}", axis=1)
        else:
            df["padded_question"] = df.apply(lambda row: f"Context: {placeholder_token * row['compressed_len']}\nQuestion: {row['question']}", axis=1)
    else:
        if "context_len" not in df.columns:
            df["context_len"] = df["context"].apply(lambda x: len(tokenizer_decoder.encode(x)))
        df["padded_question"] = df.apply(lambda row: f"Context: {row['context']}\nQuestion: {row['question']}", axis=1)

    def _apply_chat_template(q):
        return tokenizer_decoder.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
    df["templated_prompt"] = df["padded_question"].apply(_apply_chat_template)
    df_filtered = df[(df["context_len"] >= context_min_length) & (df["context_len"] <= context_max_length)].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    if max_sample_num_per_ds is not None and len(df_filtered) > max_sample_num_per_ds:
        df_filtered = df_filtered[:max_sample_num_per_ds].reset_index(drop=True)
    return df_filtered


def run_batch_generation(df_filtered, config, gpu_ids, num_processes, worker_fn):
    if num_processes > 1:
        # 轮询分配: gpu0 -> 0,8,16,...; gpu1 -> 1,9,17,...; ...
        n = len(df_filtered)
        worker_args = []
        for i in range(num_processes):
            mask = np.arange(n) % num_processes == i
            indices_i = np.where(mask)[0]
            df_chunk = df_filtered.iloc[indices_i].copy().reset_index(drop=True)
            worker_args.append((gpu_ids[i], df_chunk.to_dict("records"), indices_i.tolist(), config))
        with mp.Pool(processes=num_processes) as pool:
            results_list = pool.map(worker_fn, worker_args)
        all_results = sorted(sum(results_list, []), key=lambda x: x["index"])
    else:
        all_results = sorted(worker_fn((gpu_ids[0], df_filtered.to_dict("records"), list(range(len(df_filtered))), config)), key=lambda x: x["index"])
    result_df = pd.DataFrame(all_results).set_index("index")
    df_filtered.loc[result_df.index, "generated_text"] = result_df["generated_text"].values
    if "num_valid_tokens" in result_df.columns:
        df_filtered.loc[result_df.index, "num_valid_tokens"] = result_df["num_valid_tokens"].values
    return df_filtered


def evaluate_and_append_result(
    df_filtered,
    dataset_name,
    bridge_model_path,
    results_jsonl_path,
    is_bridge,
    comp_ratio_or_len=None,
    response_col="generated_text",
    feature_extract_method=None,
    is_dynamic=False,
    comp_ratio_or_len_for_dynamic=None,
    extra_result_record=None,
    context_len_ranges=None,
):
    accuracy_info = acc_by_context_len(df_filtered, response_col, context_len_ranges=context_len_ranges)
    compression_info = None
    if is_bridge:
        compression_info = compression_ratio_by_context_len(
            df_filtered,
            comp_ratio_or_len=comp_ratio_or_len_for_dynamic if is_dynamic else comp_ratio_or_len,
            is_dynamic=is_dynamic,
            context_len_ranges=context_len_ranges,
        )
    result_record = {
        "dataset": dataset_name,
        "bridge_model_path": bridge_model_path,
        "comp_ratio_or_len": comp_ratio_or_len,
        "num_samples": accuracy_info["num_samples"],
        "overall_accuracy": accuracy_info["overall_accuracy"],
        "context_len_accuracies": accuracy_info["context_len_accuracies"],
        "feature_extract_method": feature_extract_method,
    }
    if compression_info:
        result_record.update(compression_info)
    if extra_result_record:
        result_record.update(extra_result_record)
    if results_jsonl_path is not None:
        pathlib.Path(results_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
    return result_record


def batch_generate_worker_ctxcomp(args):
    gpu_id, df_subset, original_indices, config = args
    import pandas as pd
    from tqdm import tqdm

    if isinstance(df_subset, list):
        df_subset = pd.DataFrame(df_subset)

    bridge_model_path = config["bridge_model_path"]
    batch_size = config["batch_size"]
    comp_ratio_or_len_override = config.get("comp_ratio_or_len_override")
    is_bridge = config.get("is_bridge", True)

    if is_bridge:
        with open(os.path.join(bridge_model_path, "config.json"), "r") as f:
            model_config = json.load(f)

        model = CtxCompModel.from_pretrained(
            bridge_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": gpu_id},
            share_base_model_inference=False
        ).eval()

        print("Share base model: ",model._shared_base_model)

        embedding_path = model_config.get("base_encoder_model_path")
        generation_path = model_config.get("base_decoder_model_path")
        if generation_path is None:
            raise ValueError(f"Bridge model missing decoder path in config: {bridge_model_path}")
        tokenizer_decoder = AutoTokenizer.from_pretrained(generation_path, trust_remote_code=True, padding_side="left")
        tokenizer_encoder = AutoTokenizer.from_pretrained(embedding_path, trust_remote_code=True, padding_side="left")
        device = model.decoder.device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            bridge_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": gpu_id},
        ).eval()
        tokenizer_decoder = AutoTokenizer.from_pretrained(bridge_model_path, trust_remote_code=True, padding_side="left")
        tokenizer_encoder = None
        device = model.device

    df_subset = df_subset.reset_index(drop=True)
    results = []

    for batch_start in tqdm(range(0, len(df_subset), batch_size), desc=f"GPU {gpu_id}",mininterval=10):
        batch_end = min(batch_start + batch_size, len(df_subset))
        batch_df = df_subset.iloc[batch_start:batch_end]
        batch_templated = batch_df["templated_prompt"].tolist()
        input_encoded = tokenizer_decoder(batch_templated, return_tensors="pt", padding=True)
        input_ids = input_encoded["input_ids"].to(device)
        attention_mask = input_encoded["attention_mask"].to(device)
        if is_bridge:
            if "context_input_ids" not in batch_df.columns:
                raise ValueError(
                    "Bridge eval requires column context_input_ids from prepare_eval_dataframe; re-run data prep or use force_recompute."
                )
            rows = [_normalize_context_input_ids_row(x) for x in batch_df["context_input_ids"]]
            pad_id = tokenizer_encoder.pad_token_id or tokenizer_encoder.eos_token_id
            context_input_ids, context_attention_mask = stack_padded_context_input_ids(rows, pad_id, device)
            eos_token_id = tokenizer_decoder.eos_token_id
            # 与旧版一致：多数 CausalLM 未设置 pad_token_id，必须为 None 时回退 eos，否则 batch generate 易异常
            pad_token_id = tokenizer_decoder.pad_token_id or tokenizer_decoder.eos_token_id
            gen_kw = dict(
                context_input_ids=context_input_ids,
                input_ids=input_ids,
                context_attention_mask=context_attention_mask,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            if comp_ratio_or_len_override is not None:
                gen_kw["comp_ratio_or_len_override"] = comp_ratio_or_len_override
            generation_outputs = model.generate(**gen_kw)
            for i, output in enumerate(generation_outputs):
                generated_text = tokenizer_decoder.decode(output, skip_special_tokens=True)
                results.append({"index": original_indices[batch_start + i], "generated_text": generated_text})
            if batch_start ==0:
                print("generated_text: ",generated_text)

        else:
            generation_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer_decoder.pad_token_id or tokenizer_decoder.eos_token_id,
            )
            input_lengths = attention_mask.sum(dim=1)
            for i in range(generation_outputs.shape[0]):
                generated_ids = generation_outputs[i, int(input_lengths[i].item()):]
                generated_text = tokenizer_decoder.decode(generated_ids, skip_special_tokens=True)
                results.append({"index": original_indices[batch_start + i], "generated_text": generated_text})
            if batch_start ==0:
                print("generated_text: ",generated_text)

    return results


if __name__ == "__main__":
    LONG_CONTEXT=False
    context_max_length = 40000 if LONG_CONTEXT else 2048
    context_min_length = 128
    batch_size = 1 if LONG_CONTEXT else 8
    CONTEXT_LEN_RANGES = [(1, 10000), (10000, 20000), (20000, 30000),
                          (30000, 40000)]  if LONG_CONTEXT else [(64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]

    eval_acc = True
    max_sample_num_per_ds = 1000
    force_recompute = True
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]#[0]#
    num_processes = len(gpu_ids)
    comp_ratio_or_len_override = None  # or list to evaluate each ratio

    bridge_model_path_list = [
        #"/share/yyj/edge_memory/semi_dynamic_soft_context_compress/SFT/training_output/ctxcomp-static-mean_pooling-contextlen=1034to64-comp=0.25-enc=Qwen3.5-0.8B-lora-dec=Qwen3.5-0.8B-lora/checkpoint-40000"

        #"/share/yyj/edge_memory/semi_dynamic_soft_context_compress/SFT/training_output_swa/ctxcomp-fusang-static-swa=2k-layers=[0,2,4]-mean_pooling-contextlen=34000to128-comp=0.25-enc=lora-dec=lora/checkpoint-6000"
        #"//share/yyj/edge_memory/semi_dynamic_soft_context_compress/SFT/training_output_swa/ctxcomp-fusang-static-swanone-mean_pooling-contextlen=34000to128-comp=0.25-enc=lora-dec=lora/checkpoint-2000"
        #"//share/yyj/edge_memory/semi_dynamic_soft_context_compress/SFT/training_output_swa/ctxcomp-static-swa2k-mean_pooling-contextlen=34000to128-comp=0.25-enc=lora-dec=lora/checkpoint-10000",
        #"/share/models/Qwen3-4B-Instruct-2507",
        #"/share/models/Qwen3-0.6B",
        #"/share/models/Qwen3.5-0.8B",
        "../SFT/training_output/ctxcomp-semi-dynamic-mean_pooling-contextlen=1300to64-ratios=0.5_0.03125-enc=Qwen3.5-0.8B-lora-dec=Qwen3.5-0.8B-lora/checkpoint-100000"

    ]

    EVAL_DATA_DIR = pathlib.Path(__file__).resolve().parent / "eval_data"
    DATASETS = {
        "hotpotqa": str(EVAL_DATA_DIR / "hotpotqa_validation_cqa.parquet"),
        "squad": str(EVAL_DATA_DIR / "squad_validation_cqa.parquet"),
        "NQ": str(EVAL_DATA_DIR / "NQ_validation_cqa.parquet"),
        "adversarialQA": str(EVAL_DATA_DIR / "adverserialqa_validation_cqa.parquet"),
    } if not LONG_CONTEXT else {"longmemeval":"/share/yyj/edge_memory/semi_dynamic_soft_context_compress/Eval/eval_data_long/longmemeval_24k.parquet"}

    dataset_names_sorted = sorted(DATASETS.keys())
    multi_dataset = len(DATASETS) > 1
    summary_jsonl_path = (
        f"eval_aggregate_ctxcomp_{'_'.join(dataset_names_sorted)}_contextlen={context_min_length}to{context_max_length}.jsonl"
        if multi_dataset
        else None
    )

    for model_idx, bridge_model_path in enumerate(bridge_model_path_list):
        print(f"\n{'='*80}\nModel {model_idx+1}/{len(bridge_model_path_list)}: {bridge_model_path}")
        with open(os.path.join(bridge_model_path, "config.json"), "r") as f:
            model_config = json.load(f)
        generation_path = model_config.get("base_decoder_model_path") or model_config.get("base_gen_model_path")
        is_bridge = generation_path is not None

        if is_bridge:
            embedding_path = model_config.get("base_encoder_model_path") or model_config.get("base_embed_model_path")
            placeholder_token_id = model_config["placeholder_token_id"]
            raw_comp_ratio_or_len = comp_ratio_or_len_override if comp_ratio_or_len_override is not None else model_config.get("comp_ratio_or_len", model_config.get("num_doc_tokens", 0.25))
            feature_extract_method = model_config.get("feature_extract_method", model_config.get("compress_method", "mean_pooling"))
            tokenizer_decoder = AutoTokenizer.from_pretrained(generation_path, trust_remote_code=True)
            tokenizer_encoder = AutoTokenizer.from_pretrained(embedding_path, trust_remote_code=True)
            placeholder_token = tokenizer_decoder.convert_ids_to_tokens(placeholder_token_id)
            rates_to_eval = list(raw_comp_ratio_or_len) if isinstance(raw_comp_ratio_or_len, (list, tuple)) else [raw_comp_ratio_or_len]
        else:
            feature_extract_method = "no-compress"
            tokenizer_decoder = AutoTokenizer.from_pretrained(bridge_model_path, trust_remote_code=True)
            tokenizer_encoder = None
            placeholder_token = None
            rates_to_eval = [None]

        for current_comp in rates_to_eval:
            model_records = []
            print(f"\n--- Evaluating comp_ratio_or_len: {current_comp} ---")
            for dataset_name, data_path in DATASETS.items():
                if not os.path.isfile(data_path):
                    print(f"Skip (file not found): {data_path}")
                    continue
                print(f"\n>>> Dataset: {dataset_name} ({data_path})")
                results_jsonl_path = (
                    None
                    if multi_dataset
                    else f"eval_ctxcomp_{dataset_name}_contextlen={context_min_length}to{context_max_length}.jsonl"
                )
                save_dir = f"eval_results/{dataset_name}"
                p = pathlib.Path(bridge_model_path)
                base_name = f"{p.parent.name}-{p.name}"
                file_name = f"{base_name}-ratio_{str(current_comp).replace('.', '_')}.parquet" if len(rates_to_eval) > 1 else f"{base_name}.parquet"
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
                        is_bridge=is_bridge,
                        placeholder_token=placeholder_token,
                        feature_extract_method=feature_extract_method,
                        comp_ratio_or_len=current_comp,
                        add_eos_token_to_context=True
                    )
                    config = {
                        "bridge_model_path": bridge_model_path,
                        "comp_ratio_or_len_override": current_comp,
                        "batch_size": batch_size,
                        "is_bridge": is_bridge,
                    }
                    df_filtered = run_batch_generation(df_filtered, config, gpu_ids, num_processes, batch_generate_worker_ctxcomp)

                if eval_acc:
                    result_record = evaluate_and_append_result(
                        df_filtered,
                        dataset_name,
                        bridge_model_path,
                        results_jsonl_path,
                        is_bridge=is_bridge,
                        comp_ratio_or_len=current_comp,
                        feature_extract_method=feature_extract_method,
                        context_len_ranges=CONTEXT_LEN_RANGES,
                    )
                    model_records.append(result_record.copy())

                if (not df_exists) or force_recompute:
                    torch.cuda.empty_cache()
                    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                    df_filtered.to_parquet(save_path)

            if model_records:
                if multi_dataset and summary_jsonl_path:
                    summary = aggregate_model_results(model_records, context_len_ranges=CONTEXT_LEN_RANGES)
                    pathlib.Path(summary_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(summary_jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

#nohup python eval_ctxcomp.py > eval_comp.log 2>&1 &