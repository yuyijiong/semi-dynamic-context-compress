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
from transformers import TrainingArguments, Trainer, AutoTokenizer, Qwen3ForCausalLM, Qwen3Model
from transformers.trainer_utils import is_main_process
from peft import LoraConfig

import sft_ctxcomp_static as ctxcomp_sft

os.environ.setdefault("WANDB_PROJECT", "SFT_ctxcomp_multiratio")

DEFAULT_COMP_RATIO_OR_LEN_OPTIONS: List[Union[float, int]] = [0.5, 0.25, 0.125, 0.0625, 0.03125]


def load_single_ratio_dataset(
    data_config: List[Dict[str, Any]],
    comp_ratio_or_len_options: List[Union[float, int]],
    tokenizer_decoder,
    tokenizer_encoder,
    num_proc: int,
    max_length: int,
    min_length: int,
    doc_max_length: int,
    doc_min_length: int,
    placeholder_token: str,
    feature_extract_method: str,
    seed: int = 0,
    exclude_sign: Optional[str] = None,
    add_compress_len_labels: bool = False,
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
        doc_max_length=doc_max_length,
        doc_min_length=doc_min_length,
        comp_ratio_or_len=max_option,
        placeholder_token=placeholder_token,
        feature_extract_method=feature_extract_method,
        seed=seed,
        exclude_sign=exclude_sign,
        add_compress_len_labels=add_compress_len_labels,
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
            context_len = len(f["doc_input_ids"])
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
    parser.add_argument("--doc_max_length", type=int, default=1300)
    parser.add_argument("--doc_min_length", type=int, default=64)
    parser.add_argument("--feature_extract_method", type=str, default="mean_pooling",
                        choices=["mean_pooling", "mean_pooling_causal", "last_tokens", "same_memory_tokens", "different_memory_tokens"])
    parser.add_argument("--encoder_training", type=str, default="lora")
    parser.add_argument("--decoder_training", type=str, default="lora")
    args = parser.parse_args()

    if torch.distributed.is_initialized():
        try:
            store = torch.distributed.get_store()
            if store is not None and hasattr(store, "set_timeout"):
                store.set_timeout(4 * 3600)
        except Exception:
            pass

    max_length = 1024
    min_length = 1
    doc_max_length = args.doc_max_length
    doc_min_length = args.doc_min_length
    placeholder_id = 151656
    memory_token_begin_id = 150000
    feature_extract_method = args.feature_extract_method
    encoder_training = args.encoder_training
    decoder_training = args.decoder_training
    comp_ratio_or_len_init = tuple(DEFAULT_COMP_RATIO_OR_LEN_OPTIONS)
    GROUP_BY_LENGTH = True

    output_dir = (
        f"./training_output/ctxcomp-multiratio-{feature_extract_method}-doclen={doc_max_length}to{doc_min_length}-"
        f"ratios={comp_ratio_or_len_init[0]}_{comp_ratio_or_len_init[-1]}-enc={encoder_training}-dec={decoder_training}"
    )
    train_args = TrainingArguments(
        report_to="wandb",
        output_dir=output_dir,
        run_name=output_dir,
        optim="adamw_torch",
        learning_rate=4e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=100,
        max_steps=200000,
        eval_steps=500,
        eval_strategy="no",
        save_steps=20000,
        save_strategy="steps",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        logging_steps=20,
        bf16=True,
        fp16=False,
        seed=0,
        save_total_limit=11,
        torch_compile=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        dataloader_drop_last=True,
    )

    data_config = [
        {"dir": "/share/yyj/edge_memory/数据合成/doc_sum_qa_shortsum_synthetic_text_128_en/Qwen3-30B-A3B-Instruct-2507__vllm", "task": "qa_and_sum", "max_files": 50, "max_samples": 1000000},
    ]

    embedding_path = "/share/models/Qwen3-Embedding-0.6B"
    generation_path = "/share/models/Qwen3-0.6B"

    tokenizer_decoder = AutoTokenizer.from_pretrained(generation_path, trust_remote_code=True)
    tokenizer_encoder = AutoTokenizer.from_pretrained(embedding_path, trust_remote_code=True, padding_side="left")
    placeholder_token = tokenizer_decoder.convert_ids_to_tokens(placeholder_id)

    print("Loading encoder from", embedding_path)
    encoder = Qwen3Model.from_pretrained(
        embedding_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()

    print("Loading decoder from", generation_path)
    decoder = Qwen3ForCausalLM.from_pretrained(
        generation_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()

    from modeling_ctxcomp import CtxCompModel

    model = CtxCompModel(
        encoder=encoder,
        decoder=decoder,
        placeholder_token_id=placeholder_id,
        memory_token_begin_id=memory_token_begin_id,
        comp_ratio_or_len=comp_ratio_or_len_init,
        mlp_converter_hidden_dim=4096,
        feature_extract_method=feature_extract_method,
        encoder_training=encoder_training,
        decoder_training=decoder_training,
    )

    emb_lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.1, bias="none", use_rslora=True)
    gen_lora_config = LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.1, bias="none", use_rslora=True, task_type="CAUSAL_LM")
    model.set_trainable_params(emb_lora_config, gen_lora_config)
    model.reinit_memory_tokens()
    model.print_trainable_parameters()

    with train_args.main_process_first(desc="Loading single-ratio dataset (max ratio)"):
        num_proc = 4
        ds = load_single_ratio_dataset(
            data_config=data_config,
            comp_ratio_or_len_options=DEFAULT_COMP_RATIO_OR_LEN_OPTIONS,
            tokenizer_decoder=tokenizer_decoder,
            tokenizer_encoder=tokenizer_encoder,
            num_proc=num_proc,
            max_length=max_length,
            min_length=min_length,
            doc_max_length=doc_max_length,
            doc_min_length=doc_min_length,
            placeholder_token=placeholder_token,
            feature_extract_method=feature_extract_method,
            seed=train_args.seed,
            exclude_sign="eval",
            generation_path=generation_path,
            template_dir=".",
        )
        print("Dataset size (max ratio):", len(ds))

    data_collator = MultiratioCtxCompCollator(
        comp_ratio_or_len_options=DEFAULT_COMP_RATIO_OR_LEN_OPTIONS,
        placeholder_id=placeholder_id,
        feature_extract_method=feature_extract_method,
        tokenizer=tokenizer_decoder,
        tokenizer_encoder=tokenizer_encoder,
        padding=True,
        max_length=doc_max_length,
        label_pad_token_id=-100,
    )

    if GROUP_BY_LENGTH:
        trainer = ctxcomp_sft.BucketedCtxCompTrainer(
            model=model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=ds,
            eval_dataset=None,
            processing_class=tokenizer_decoder,
            length_column_name="context_len",
            num_buckets=5,
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
