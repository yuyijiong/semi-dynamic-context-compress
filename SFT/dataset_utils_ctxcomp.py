"""
Dataset processing utilities for CtxComp SFT.

Provides: chat template setup, encode_* functions, load_from_data_config.
Used by sft_ctxcomp_static.py, sft_ctxcomp_static_swa.py, sft_ctxcomp_static_multiratio.py, etc.

When using return_assistant_tokens_mask=True with apply_chat_template, the tokenizer's
chat_template must contain `{% generation %}` (e.g. Qwen3 non-thinking template).
apply_qwen3_chat_template() is called at the start of load_from_data_config when
generation_path is provided, so it is a fixed step before any dataset encoding.
"""
import os
import random
import math
from typing import Optional, Any, List, Dict, Union, Sequence

import numpy as np
from datasets import Dataset, concatenate_datasets


def apply_qwen3_chat_template(tokenizer, generation_path: str, template_dir: Optional[str] = None) -> None:
    """
    Set tokenizer.chat_template from the appropriate Qwen3 template file so that
    apply_chat_template(..., return_assistant_tokens_mask=True) works (template must
    contain `{% generation %}`). Call this before any dataset encoding that uses
    return_assistant_tokens_mask.

    Args:
        tokenizer: Decoder tokenizer to modify.
        generation_path: Model path or name; used to choose instruct/thinking/nonthink.
        template_dir: Directory containing qwen3-instruct-template, qwen3-thinking-template,
            qwen3-nonthink-template. Default "." (current working directory).
    """
    if template_dir is None:
        template_dir = "."
    base = os.path.join(template_dir, "qwen3-{}-template")
    if "instruct" in generation_path.lower():
        path = base.format("instruct")
        print("使用instruct模板")
    elif "thinking" in generation_path.lower():
        path = base.format("thinking")
        print("使用thinking模板")
    else:
        path = base.format("nonthink")
        print("使用nonthink模板")
    with open(path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()


def _is_ratio_based_method(method: str) -> bool:
    return method in ("mean_pooling", "mean_pooling_causal")


def _n_placeholders_from_ratio(
    context_len: int, comp_ratio_or_len: Union[int, float], feature_extract_method: str
) -> int:
    if (
        _is_ratio_based_method(feature_extract_method)
        and isinstance(comp_ratio_or_len, (int, float))
        and 0 < comp_ratio_or_len <= 1
    ):
        pool_size = max(1, round(1.0 / float(comp_ratio_or_len)))
        return max(1, (context_len + pool_size - 1) // pool_size)
    return max(1, min(int(comp_ratio_or_len), context_len))


def encode_qa_and_sum(
    example,
    tokenizer_decoder,
    pretrain=False,
    system_prompt=None,
    num_doc_tokens=0,
    placeholder_token="<|endoftext|>",
    feature_extract_method="mean_pooling",
):
    context_len = len(example["doc_input_ids"])
    n_placeholders = _n_placeholders_from_ratio(context_len, num_doc_tokens, feature_extract_method)

    has_qa = "question" in example and "answer" in example and example.get("question") and example.get("answer")
    has_sum = "summary" in example and example.get("summary")

    if has_qa and has_sum:
        if random.random() < 0.5:
            first_sum_prompt = f"Context: {placeholder_token * n_placeholders}\n\nPlease write a concise summary of the above context."
            conversations = [
                {"role": "user", "content": first_sum_prompt},
                {"role": "assistant", "content": example["summary"]},
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        else:
            conversations = [
                {"role": "user", "content": f"Context: {placeholder_token * n_placeholders}\n\nQuestion: {example['question']}"},
                {"role": "assistant", "content": example["answer"]},
                {"role": "user", "content": "Write a concise summary of the above context."},
                {"role": "assistant", "content": example["summary"]},
            ]
    elif has_qa:
        user_content = f"Context: {placeholder_token * n_placeholders}\n\nQuestion: {example['question']}"
        conversations = [{"role": "user", "content": user_content}, {"role": "assistant", "content": example["answer"]}]
    elif has_sum:
        user_content = f"Context: {placeholder_token * n_placeholders}\n\nPlease write a concise summary of the above context."
        conversations = [{"role": "user", "content": user_content}, {"role": "assistant", "content": example["summary"]}]
    else:
        raise ValueError("encode_qa_and_sum: example must have at least one of (question+answer) or summary")

    if system_prompt is not None:
        conversations.insert(0, {"role": "system", "content": system_prompt})

    ids = tokenizer_decoder.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
        truncation=False,
    )
    labels = ids["input_ids"].copy()
    if not pretrain:
        for i, mask in enumerate(ids["assistant_masks"]):
            if mask == 0:
                labels[i] = -100
    return {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"], "labels": labels}


def encode_multi_doc_qa(
    example,
    tokenizer_decoder,
    pretrain=False,
    system_prompt=None,
    num_doc_tokens=0,
    placeholder_token="<|endoftext|>",
    feature_extract_method="mean_pooling",
):
    context_len = len(example["doc_input_ids"])
    n_placeholders = _n_placeholders_from_ratio(context_len, num_doc_tokens, feature_extract_method)
    questions = example["questions"]
    answers = example["answers"]
    if not questions or not answers or len(questions) != len(answers):
        raise ValueError("encode_multi_doc_qa: questions and answers must be non-empty and same length")
    first_user_content = f"Context: {placeholder_token * n_placeholders}\n\nQuestion: {questions[0]}"
    conversations = [{"role": "user", "content": first_user_content}, {"role": "assistant", "content": answers[0]}]
    for q, a in zip(questions[1:], answers[1:]):
        conversations.append({"role": "user", "content": q})
        conversations.append({"role": "assistant", "content": a})
    if system_prompt is not None:
        conversations.insert(0, {"role": "system", "content": system_prompt})
    ids = tokenizer_decoder.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
        truncation=False,
    )
    labels = ids["input_ids"].copy()
    if not pretrain:
        for i, mask in enumerate(ids["assistant_masks"]):
            if mask == 0:
                labels[i] = -100
    return {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"], "labels": labels}


def encode_text_reconstruction(
    example,
    tokenizer_decoder,
    pretrain=False,
    system_prompt=None,
    num_doc_tokens=0,
    placeholder_token="<|endoftext|>",
    feature_extract_method="mean_pooling",
):
    context_len = len(example["doc_input_ids"])
    n_placeholders = _n_placeholders_from_ratio(context_len, num_doc_tokens, feature_extract_method)
    conversations = [
        {"role": "user", "content": f"Context: {placeholder_token * n_placeholders}\n\nPlease repeat the context."},
        {"role": "assistant", "content": example["context"]},
    ]
    if system_prompt is not None:
        conversations.insert(0, {"role": "system", "content": system_prompt})
    ids = tokenizer_decoder.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
        truncation=False,
    )
    labels = ids["input_ids"].copy()
    if not pretrain:
        for i, mask in enumerate(ids["assistant_masks"]):
            if mask == 0:
                labels[i] = -100
    return {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"], "labels": labels}


def encode_doc(example, tokenizer_encoder, add_compress_len_labels: bool = False):
    ids = tokenizer_encoder(example["context"])
    out = {
        "doc_input_ids": ids["input_ids"],
        "doc_attention_mask": ids["attention_mask"],
        "context_len": len(ids["input_ids"]),
    }
    if add_compress_len_labels and "short_summary" in example and example.get("short_summary"):
        short_ids = tokenizer_encoder(example["short_summary"], add_special_tokens=False)
        short_len = len(short_ids["input_ids"])
        out["compress_len_labels"] = math.log2(len(ids["input_ids"]) / short_len) if short_len else 0.0
    return out


def build_multi_doc_context(example):
    if "docs" not in example:
        raise ValueError("docs not in dataset")
    docs = example["docs"]
    indices = list(range(len(docs)))
    random.shuffle(indices)
    parts = [f"Passage-{i + 1}\n{docs[idx]}" for i, idx in enumerate(indices)]
    return {**example, "context": "\n\n".join(parts)}


def get_file_paths(dir, suffix, subfolder=True, exclude_suffix=None):
    file_path_list = []
    if not subfolder:
        for file in os.listdir(dir):
            if file.endswith(suffix):
                file_path_list.append(os.path.join(dir, file))
    else:
        for root, _dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(suffix):
                    file_path_list.append(os.path.join(root, file))
    if exclude_suffix is not None:
        file_path_list = [p for p in file_path_list if not p.endswith(exclude_suffix)]
    return file_path_list


def get_sorted_files_from_dir(dir_path, suffix=".parquet", subfolder=True, exclude_sign=None, max_files=None):
    files = get_file_paths(dir_path, suffix, subfolder=subfolder)
    files.sort()
    if exclude_sign is not None:
        files = [f for f in files if exclude_sign not in f]
    if max_files is not None:
        files = files[:max_files]
    return files


def rename_context(ds):
    for col in ["text_256", "text_512", "text_1024", "text_2048", "text_128"]:
        if 'context' in ds.column_names:
            return ds
        if col in ds.column_names:
            return ds.rename_column(col, "context")
    raise ValueError("No text_* column found for context.")


def load_from_data_config(
    data_config: List[Dict[str, Any]],
    tokenizer_decoder,
    tokenizer_encoder,
    num_proc: int,
    max_length: int,
    min_length: int,
    doc_max_length: int,
    doc_min_length: int,
    comp_ratio_or_len: Union[int, float],
    placeholder_token: str,
    feature_extract_method: str,
    seed: int = 0,
    exclude_sign: Optional[str] = None,
    add_compress_len_labels: bool = False,
    generation_path: Optional[str] = None,
    template_dir: Optional[str] = None,
) -> Dataset:
    """
    Load and encode dataset from data_config. Supported tasks: qa_and_sum, qa, sum,
    multi_doc_qa, text_reconstruct.

    If generation_path is not None, applies Qwen3 chat template (instruct/thinking/nonthink)
    to tokenizer_decoder at the start, so that return_assistant_tokens_mask works.
    """
    if generation_path is not None:
        apply_qwen3_chat_template(tokenizer_decoder, generation_path, template_dir)

    supported_tasks = ("qa_and_sum", "qa", "sum", "multi_doc_qa", "text_reconstruct")
    empty_dict = {"input_ids": [], "attention_mask": [], "labels": [], "doc_input_ids": [], "doc_attention_mask": []}
    if add_compress_len_labels:
        empty_dict["compress_len_labels"] = []
    ds_list = []

    for item in data_config:
        dir_or_file = item["dir"]
        task = item["task"]
        max_files = item.get("max_files")
        max_samples = item.get("max_samples")
        if task not in supported_tasks:
            raise ValueError(f"Unknown task: {task}")

        if os.path.isfile(dir_or_file):
            files = [dir_or_file]
        else:
            files = get_sorted_files_from_dir(dir_or_file, suffix=".parquet", subfolder=True, exclude_sign=exclude_sign)
            if max_files is not None:
                files = files[:max_files]
        if not files:
            continue

        ds_ori = concatenate_datasets([Dataset.from_parquet(f) for f in files])
        if max_samples is not None and len(ds_ori) > 0:
            ds_ori = ds_ori.select(range(min(max_samples, len(ds_ori))))

        if task == "multi_doc_qa" and "context" not in ds_ori.column_names:
            ds_ori = ds_ori.map(build_multi_doc_context, num_proc=num_proc, remove_columns=["docs"])
        else:
            ds_ori = rename_context(ds_ori)

        ds_doc = ds_ori.map(
            lambda e: encode_doc(e, tokenizer_encoder=tokenizer_encoder, add_compress_len_labels=add_compress_len_labels),
            num_proc=num_proc,
            desc="Encoding doc",
        )
        if add_compress_len_labels:

            def _ensure_compress_len_labels(e):
                if "compress_len_labels" not in e or e.get("compress_len_labels") is None:
                    return {**e, "compress_len_labels": 0.0}
                return e

            ds_doc = ds_doc.map(_ensure_compress_len_labels, num_proc=num_proc, desc="Ensure compress_len_labels")
        ds_doc = ds_doc.filter(lambda x: doc_min_length <= x["context_len"] <= doc_max_length, num_proc=num_proc)
        if len(ds_doc) == 0:
            continue

        keep_cols = ["doc_input_ids", "doc_attention_mask", "context_len"] + (
            ["compress_len_labels"] if add_compress_len_labels else []
        )

        def _encode_gen(ds, ratio, desc_suffix=""):
            if task == "multi_doc_qa":
                ds_out = ds.map(
                    lambda e: encode_multi_doc_qa(
                        e,
                        tokenizer_decoder=tokenizer_decoder,
                        num_doc_tokens=ratio,
                        placeholder_token=placeholder_token,
                        feature_extract_method=feature_extract_method,
                    ),
                    remove_columns=[c for c in ds.column_names if c not in keep_cols],
                    num_proc=num_proc,
                    desc=f"multi_doc_qa {ratio}{desc_suffix}",
                )
            elif task == "text_reconstruct":
                ds_out = ds.map(
                    lambda e: encode_text_reconstruction(
                        e,
                        tokenizer_decoder=tokenizer_decoder,
                        num_doc_tokens=ratio,
                        placeholder_token=placeholder_token,
                        feature_extract_method=feature_extract_method,
                    ),
                    remove_columns=[c for c in ds.column_names if c not in keep_cols],
                    num_proc=num_proc,
                    desc=f"text_reconstruct {ratio}{desc_suffix}",
                )
            else:
                ds_out = ds.map(
                    lambda e: encode_qa_and_sum(
                        e,
                        tokenizer_decoder=tokenizer_decoder,
                        num_doc_tokens=ratio,
                        placeholder_token=placeholder_token,
                        feature_extract_method=feature_extract_method,
                    ),
                    remove_columns=[c for c in ds.column_names if c not in keep_cols],
                    num_proc=num_proc,
                    desc=f"{task} {ratio}{desc_suffix}",
                )
            return ds_out.filter(lambda x: min_length <= len(x["input_ids"]) <= max_length, num_proc=num_proc)

        result = _encode_gen(ds_doc, comp_ratio_or_len)
        if len(result) > 0:
            ds_list.append(result)

    if not ds_list:
        return Dataset.from_dict(empty_dict)
    return concatenate_datasets(ds_list)
