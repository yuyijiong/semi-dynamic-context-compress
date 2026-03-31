"""
SFT script for CtxCompModel with multiple compression ratios (static multiratio).

Single dataset built with max ratio/length; data collator randomly picks a ratio per batch
and shortens placeholders accordingly. Batch key comp_ratio_or_len_override is passed to model forward.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import random
import math
import argparse
import pathlib
from typing import List, Dict, Any, Optional, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Model,AutoModelForCausalLM,AutoModel
from transformers.trainer_utils import is_main_process
from peft import LoraConfig

import sft_ctxcomp_static as ctxcomp_sft
from sft_ctxcomp_train_config import (
    add_ctxcomp_sft_cli_args,
    build_ctxcomp_training_arguments,
    comp_ratio_or_len_as_tuple,
    format_comp_ratio_for_output_dir,
)
from dataset_utils_ctxcomp import DEFAULT_DATA_CONFIG

os.environ.setdefault("WANDB_PROJECT", "SFT_ctxcomp_multiratio")

MULTIRATIO_SFT_CLI_DEFAULTS = {
    "comp_ratio_or_len": (0.5, 0.25, 0.125, 0.0625, 0.03125),
    "encoder_base_model_path": "/share/models/Qwen3-Embedding-0.6B",
    "decoder_base_model_path": "/share/models/Qwen3-0.6B",
    "mlp_converter_hidden_dim": 4096,
    "placeholder_id": 151656,
    "memory_token_begin_id": 150000,
    "feature_extract_method": "mean_pooling",
    "encoder_training": "lora",
    "decoder_training": "lora",
    "context_max_length": 1300,
    "context_min_length": 64,
    "decoder_max_length": 1024,
    "decoder_min_length": 1,
    "dataset_num_proc": 4,
    "dataset_exclude_sign": "eval",
    "template_dir": ".",
    "add_eos_token_to_context": True,
    "use_gradient_checkpointing": False,
    "per_device_train_batch_size": 8,
    "max_steps": 200000,
    "gradient_accumulation_steps": 1,
    "learning_rate": 4e-5,
    "warmup_steps": 100,
    "eval_steps": 500,
    "save_steps": 20000,
    "save_total_limit": 11,
    "logging_steps": 20,
    "lr_scheduler_type": "constant_with_warmup",
    "seed": 0,
    "group_by_length": True,
    "num_buckets": 5,
    "collator_pad_to_multiple_of": None,
}


def load_single_ratio_dataset(
    data_config: List[Dict[str, Any]],
    comp_ratio_or_len_options: List[Union[float, int]],
    tokenizer_decoder,
    tokenizer_encoder,
    num_proc: int,
    max_length: int,
    min_length: int,
    context_max_length: int,
    context_min_length: int,
    placeholder_token: str,
    feature_extract_method: str,
    seed: int = 0,
    exclude_sign: Optional[str] = None,
    add_compress_len_labels: bool = False,
    add_eos_token_to_context: bool = True,
    generation_path: Optional[str] = None,
    template_dir: Optional[str] = None,
) -> Dataset:
    """Build dataset with max comp_ratio_or_len only; collator will vary ratio per batch."""
    if not comp_ratio_or_len_options:
        raise ValueError("comp_ratio_or_len_options must be non-empty")
    max_option = max(comp_ratio_or_len_options)
    return ctxcomp_sft.load_from_data_config(
        data_config=data_config,
        tokenizer_decoder=tokenizer_decoder,
        tokenizer_encoder=tokenizer_encoder,
        num_proc=num_proc,
        max_length=max_length,
        min_length=min_length,
        context_max_length=context_max_length,
        context_min_length=context_min_length,
        comp_ratio_or_len=max_option,
        placeholder_token=placeholder_token,
        feature_extract_method=feature_extract_method,
        seed=seed,
        exclude_sign=exclude_sign,
        add_compress_len_labels=add_compress_len_labels,
        add_eos_token_to_context=add_eos_token_to_context,
        generation_path=generation_path,
        template_dir=template_dir,
    )


def _placeholder_span(ids: List[int], placeholder_id: int) -> Optional[tuple]:
    indices = [i for i, tid in enumerate(ids) if tid == placeholder_id]
    if not indices:
        return None
    return min(indices), max(indices) + 1


def _shorten_sequence_by_span(seq: List[int], start: int, end: int, target_n: int, fill_id: int) -> List[int]:
    current_n = end - start
    if target_n >= current_n:
        return seq
    return seq[:start] + [fill_id] * target_n + seq[end:]


def _target_n_placeholders(context_len: int, comp_ratio_or_len: Union[float, int], feature_extract_method: Optional[str] = None) -> int:
    """Number of placeholders to use; must match model's M for mean_pooling (pool_size=round(1/ratio), M=ceil(L/pool_size))."""
    if isinstance(comp_ratio_or_len, float) and 0 < comp_ratio_or_len <= 1:
        if feature_extract_method in ("mean_pooling", "mean_pooling_causal"):
            pool_size = max(1, round(1.0 / comp_ratio_or_len))
            return max(1, (context_len + pool_size - 1) // pool_size)
        return max(1, math.ceil(context_len * comp_ratio_or_len))
    return max(1, min(int(comp_ratio_or_len), context_len))


class MultiratioCtxCompCollator(ctxcomp_sft.CtxCompDataCollator):
    """Picks one comp_ratio_or_len per batch, shortens placeholders, sets comp_ratio_or_len_override."""

    def __init__(
        self,
        comp_ratio_or_len_options: List[Union[float, int]],
        placeholder_id: int,
        feature_extract_method: Optional[str] = None,
        ratios_weight: Optional[List[float]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.comp_ratio_or_len_options = comp_ratio_or_len_options
        self.placeholder_id = placeholder_id
        self.feature_extract_method = feature_extract_method
        if ratios_weight is None:
            self._weights = [1.0] * len(comp_ratio_or_len_options)
        else:
            if len(ratios_weight) != len(comp_ratio_or_len_options):
                raise ValueError("ratios_weight length must match comp_ratio_or_len_options")
            self._weights = list(ratios_weight)

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if not features:
            return super().__call__(features, return_tensors=return_tensors)

        chosen = random.choices(self.comp_ratio_or_len_options, weights=self._weights, k=1)[0]
        processed = []
        for f in features:
            context_len = len(f["context_input_ids"])
            target_n = _target_n_placeholders(context_len, chosen, self.feature_extract_method)
            input_ids = list(f["input_ids"])
            attention_mask = list(f["attention_mask"])
            labels = list(f["labels"])
            span = _placeholder_span(input_ids, self.placeholder_id)
            if span is not None:
                start, end = span
                input_ids = _shorten_sequence_by_span(input_ids, start, end, target_n, self.placeholder_id)
                attention_mask = _shorten_sequence_by_span(attention_mask, start, end, target_n, 1)
                labels = _shorten_sequence_by_span(labels, start, end, target_n, -100)
            processed.append({**f, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

        batch = super().__call__(processed, return_tensors=return_tensors)
        batch["comp_ratio_or_len_override"] = chosen
        return batch


def main():
    parser = argparse.ArgumentParser(description="Train CtxCompModel (multi-ratio)")
    add_ctxcomp_sft_cli_args(parser, MULTIRATIO_SFT_CLI_DEFAULTS)
    args = parser.parse_args()

    if torch.distributed.is_initialized():
        try:
            store = torch.distributed.get_store()
            if store is not None and hasattr(store, "set_timeout"):
                store.set_timeout(4 * 3600)
        except Exception:
            pass

    comp_ratio_options = comp_ratio_or_len_as_tuple(args.comp_ratio_or_len)

    output_dir = (
        f"./training_output/ctxcomp-multiratio-{args.feature_extract_method}-contextlen={args.context_max_length}to{args.context_min_length}-"
        f"ratios={format_comp_ratio_for_output_dir(args.comp_ratio_or_len)}-enc={args.encoder_training}-dec={args.decoder_training}"
    )
    train_args = build_ctxcomp_training_arguments(
        args,
        output_dir=output_dir,
        ddp_find_unused_parameters=not args.use_gradient_checkpointing,
        gradient_checkpointing=False,
    )

    data_config = DEFAULT_DATA_CONFIG

    tokenizer_decoder = AutoTokenizer.from_pretrained(args.decoder_base_model_path, trust_remote_code=True)
    tokenizer_encoder = AutoTokenizer.from_pretrained(args.encoder_base_model_path, trust_remote_code=True, padding_side="left")
    placeholder_token = tokenizer_decoder.convert_ids_to_tokens(args.placeholder_id)

    print("Loading encoder from", args.encoder_base_model_path)
    encoder = AutoModel.from_pretrained(
        args.encoder_base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()

    print("Loading decoder from", args.decoder_base_model_path)
    decoder = AutoModelForCausalLM.from_pretrained(
        args.decoder_base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()

    if args.use_gradient_checkpointing:
        encoder.gradient_checkpointing_enable({"use_reentrant": False})
        decoder.gradient_checkpointing_enable({"use_reentrant": False})

    from modeling_ctxcomp import CtxCompModel

    model = CtxCompModel(
        encoder=encoder,
        decoder=decoder,
        placeholder_token_id=args.placeholder_id,
        memory_token_begin_id=args.memory_token_begin_id,
        comp_ratio_or_len=comp_ratio_options,
        mlp_converter_hidden_dim=args.mlp_converter_hidden_dim,
        feature_extract_method=args.feature_extract_method,
        encoder_training=args.encoder_training,
        decoder_training=args.decoder_training,
    )

    emb_lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.1, bias="none", use_rslora=True)
    gen_lora_config = LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.1, bias="none", use_rslora=True, task_type="CAUSAL_LM")
    model.set_trainable_params(emb_lora_config, gen_lora_config)
    model.reinit_memory_tokens()
    model.print_trainable_parameters()

    with train_args.main_process_first(desc="Loading single-ratio dataset (max ratio)"):
        ds = load_single_ratio_dataset(
            data_config=data_config,
            comp_ratio_or_len_options=list(comp_ratio_options),
            tokenizer_decoder=tokenizer_decoder,
            tokenizer_encoder=tokenizer_encoder,
            num_proc=args.dataset_num_proc,
            max_length=args.decoder_max_length,
            min_length=args.decoder_min_length,
            context_max_length=args.context_max_length,
            context_min_length=args.context_min_length,
            placeholder_token=placeholder_token,
            feature_extract_method=args.feature_extract_method,
            seed=train_args.seed,
            exclude_sign=args.dataset_exclude_sign or None,
            add_eos_token_to_context=args.add_eos_token_to_context,
            generation_path=args.decoder_base_model_path,
            template_dir=args.template_dir,
        )
        print("Dataset size (max ratio):", len(ds))

    _collator_kw = {}
    if args.collator_pad_to_multiple_of is not None:
        _collator_kw["pad_to_multiple_of"] = args.collator_pad_to_multiple_of
    data_collator = MultiratioCtxCompCollator(
        comp_ratio_or_len_options=list(comp_ratio_options),
        placeholder_id=args.placeholder_id,
        feature_extract_method=args.feature_extract_method,
        tokenizer=tokenizer_decoder,
        tokenizer_encoder=tokenizer_encoder,
        padding=True,
        max_length=args.context_max_length,
        label_pad_token_id=-100,
        **_collator_kw,
    )

    if args.group_by_length:
        trainer = ctxcomp_sft.BucketedCtxCompTrainer(
            model=model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=ds,
            eval_dataset=None,
            processing_class=tokenizer_decoder,
            length_column_name="context_len",
            num_buckets=args.num_buckets,
        )
    else:
        trainer = ctxcomp_sft.CtxCompTrainer(
            model=model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=ds,
            eval_dataset=None,
            processing_class=tokenizer_decoder,
        )

    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    main()
