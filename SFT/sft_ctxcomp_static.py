"""
SFT script for CtxCompModel (static single compression ratio/length).

Uses modeling_ctxcomp.CtxCompModel. Compatible with feature_extract_method:
mean_pooling, mean_pooling_causal, last_tokens, same_memory_tokens, different_memory_tokens.
"""
import os
os.environ.setdefault("WANDB_PROJECT", "SFT_ctxcomp")
os.environ.setdefault("TORCH_DISTRIBUTED_DEFAULT_TIMEOUT", "14400")

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import (
    Trainer,
    AutoTokenizer,
    Qwen3ForCausalLM,
    Qwen3Model,
AutoModelForCausalLM,AutoModel
)
from transformers.trainer_utils import is_main_process
from peft import LoraConfig
from dataclasses import dataclass
from typing import Optional, Any, List, Dict, Union, Sequence, Tuple
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForSeq2Seq
from torch.utils.data import BatchSampler
import hashlib
import json
import pathlib
import argparse
import numpy as np
from datetime import timedelta
from datasets.utils.logging import set_verbosity_error, disable_progress_bar

#set_verbosity_error()
#disable_progress_bar()

from modeling_ctxcomp import CtxCompModel
from sft_ctxcomp_train_config import (
    add_ctxcomp_sft_cli_args,
    build_ctxcomp_training_arguments,
    format_comp_ratio_for_output_dir,
)
from dataset_utils_ctxcomp import (
    DEFAULT_DATA_CONFIG,
    load_from_data_config,
    encode_qa_and_sum,
    encode_multi_context_qa,
    encode_text_reconstruction,
    encode_context,
    build_context_from_passages,
    rename_context,
)

_COLLATOR_EXCLUDE_KEYS = frozenset({"context_len"})
_COLLATOR_CONTEXT_ENCODER_KEYS = frozenset({"context_input_ids", "context_attention_mask"})


@dataclass
class CtxCompDataCollator(DataCollatorForSeq2Seq):
    """Collator for CtxComp: gen part right-padded, context encoder part left-padded; passes comp_ratio_or_len_override when present."""
    tokenizer_encoder: PreTrainedTokenizerBase = None

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        gen_features = []
        context_encoder_features = []
        for feature in features:
            gen_f = {
                k: v
                for k, v in feature.items()
                if k not in _COLLATOR_CONTEXT_ENCODER_KEYS and k not in _COLLATOR_EXCLUDE_KEYS
            }
            gen_features.append(gen_f)
            ctx_pad_item = {"input_ids": feature["context_input_ids"], "attention_mask": feature["context_attention_mask"]}
            context_encoder_features.append(ctx_pad_item)

        batch = super().__call__(gen_features, return_tensors=return_tensors)

        if self.tokenizer_encoder is None:
            raise ValueError("CtxCompDataCollator requires tokenizer_encoder.")
        context_padded = self.tokenizer_encoder.pad(
            context_encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        batch["context_input_ids"] = context_padded["input_ids"]
        batch["context_attention_mask"] = context_padded["attention_mask"]

        #确保已经pad_to_multiple_of
        #assert batch["context_input_ids"].shape[1] % self.pad_to_multiple_of == 0

        if features and "comp_ratio_or_len_override" in features[0]:
            batch["comp_ratio_or_len_override"] = features[0]["comp_ratio_or_len_override"]

        for key in _COLLATOR_EXCLUDE_KEYS:
            batch.pop(key, None)
        return batch


def _dataset_context_lens_cache_key(dataset, length_column_name: str) -> Optional[str]:
    """Key for .train_context_lens cache; changes when dataset identity / HF fingerprint changes.
    Returns None if a stable key cannot be built (skip disk cache to avoid stale hits)."""
    parts = [str(len(dataset)), length_column_name]
    fp = getattr(dataset, "fingerprint", None)
    if fp is not None:
        parts.append(str(fp))
    elif getattr(dataset, "_fingerprint", None) is not None:
        parts.append(str(dataset._fingerprint))
    else:
        cfs = getattr(dataset, "cache_files", None) or []
        if cfs:
            parts.append(json.dumps(cfs, sort_keys=True, default=str))
        else:
            return None
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def build_buckets_from_quantiles(lengths: Sequence[int], num_buckets: int = 24) -> List[int]:
    if num_buckets < 2:
        return [max(lengths)] if lengths else []
    n = len(lengths)
    if n == 0:
        return []
    arr = np.asarray(lengths, dtype=np.int64)
    qs = np.linspace(0, 100, num_buckets + 1)[1:-1]
    edges = np.percentile(arr, qs).astype(np.int64).tolist()
    return edges + [int(arr.max())]


def bucket_index(length: int, edges: Sequence[int]) -> int:
    lo, hi = 0, len(edges) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if length <= edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_edges: Sequence[int],
        drop_last: bool = False,
        process_rank: int = 0,
        num_processes: int = 1,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_edges = list(bucket_edges)
        self.drop_last = drop_last
        self.process_rank = process_rank
        self.num_processes = max(1, num_processes)
        self.generator = generator
        self.seed = seed
        self._buckets = [[] for _ in range(len(self.bucket_edges))]
        for idx, L in enumerate(self.lengths):
            b = bucket_index(L, self.bucket_edges)
            self._buckets[b].append(idx)

    def __iter__(self):
        g = self.generator
        if g is None:
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed)
            else:
                g.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

        batch_lists_per_bucket = []
        for bucket in self._buckets:
            bucket_indices = bucket[:]
            if bucket_indices:
                perm = torch.randperm(len(bucket_indices), generator=g).tolist()
                bucket_indices = [bucket_indices[i] for i in perm]
            batches_b = []
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    batches_b.append(batch)
            batch_lists_per_bucket.append(batches_b)

        N = self.num_processes
        if N > 1:
            for i in range(len(batch_lists_per_bucket)):
                n = len(batch_lists_per_bucket[i])
                n_rounds = n // N
                batch_lists_per_bucket[i] = batch_lists_per_bucket[i][: n_rounds * N]

        rounds = []
        for b, batches_b in enumerate(batch_lists_per_bucket):
            for k in range(0, len(batches_b), N):
                rounds.append((b, k))
        if rounds:
            perm = torch.randperm(len(rounds), generator=g).tolist()
            rounds = [rounds[i] for i in perm]

        ordered = []
        for b, k in rounds:
            for r in range(N):
                ordered.append(batch_lists_per_bucket[b][k + r])

        if N > 1:
            ordered = ordered[self.process_rank :: N]

        for b in ordered:
            yield b

    def __len__(self):
        total = 0
        for bucket in self._buckets:
            n = len(bucket) // self.batch_size
            if not self.drop_last and (len(bucket) % self.batch_size):
                n += 1
            if self.num_processes > 1:
                n = (n // self.num_processes) * self.num_processes
            total += n
        if self.num_processes > 1:
            return total // self.num_processes
        return total


def _is_ctxcomp_checkpoint(checkpoint_dir: str) -> bool:
    config_path = pathlib.Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return False
    if config.get("base_encoder_model_path") is not None or config.get("base_embed_model_path") is not None:
        return True
    return pathlib.Path(checkpoint_dir).joinpath("encoder").is_dir() or pathlib.Path(checkpoint_dir).joinpath("embed_model").is_dir()


class CtxCompTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is not None:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            if self.is_world_process_zero():
                self.model.save_pretrained(output_dir)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        if model is None:
            model = self.model
        if not _is_ctxcomp_checkpoint(resume_from_checkpoint):
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        import logging
        logging.getLogger("transformers.trainer").info(
            f"Loading CtxComp model from custom checkpoint: {resume_from_checkpoint}"
        )
        self.model.load_from_checkpoint(resume_from_checkpoint)

    def _load_optimizer_and_scheduler(self, resume_from_checkpoint: str):
        try:
            super()._load_optimizer_and_scheduler(resume_from_checkpoint)
        except ValueError as e:
            if "parameter group" in str(e) and "doesn't match" in str(e):
                import logging
                logging.getLogger("transformers.trainer").warning(
                    f"Skip loading optimizer/scheduler: {e}. Training continues with fresh optimizer."
                )
            else:
                raise


class BucketedCtxCompTrainer(CtxCompTrainer):
    def __init__(self, *args, length_column_name: str = "context_len", num_buckets: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_column_name = length_column_name
        self.num_buckets = num_buckets

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return super().get_train_dataloader()
        if hasattr(self.train_dataset, "with_format"):
            self.train_dataset = self.train_dataset.with_format("torch")

        process_rank = getattr(self.args, "process_index", 0)
        num_processes = getattr(self.args, "world_size", 1)
        use_dist = num_processes > 1 and torch.distributed.is_initialized()
        cache_dir = getattr(self.args, "output_dir", None)
        cache_key = _dataset_context_lens_cache_key(self.train_dataset, self.length_column_name)
        cache_path = (
            pathlib.Path(cache_dir, f".train_context_lens.{cache_key}.npy")
            if cache_dir and cache_key is not None
            else None
        )

        if cache_path and cache_path.is_file():
            lengths = np.load(cache_path).tolist()
        else:
            if use_dist:
                if process_rank == 0:
                    lengths = self.train_dataset[self.length_column_name]
                    if isinstance(lengths, torch.Tensor):
                        lengths = lengths.tolist()
                    pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_path, np.asarray(lengths))
                torch.distributed.barrier()
                if process_rank != 0:
                    lengths = np.load(cache_path).tolist()
            else:
                lengths = self.train_dataset[self.length_column_name]
                if isinstance(lengths, torch.Tensor):
                    lengths = lengths.tolist()
                if cache_path:
                    pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_path, np.asarray(lengths))

        edges = build_buckets_from_quantiles(lengths, num_buckets=self.num_buckets)
        seed = getattr(self.args, "seed", 42)
        batch_sampler = BucketBatchSampler(
            lengths=lengths,
            batch_size=self.args.per_device_train_batch_size,
            bucket_edges=edges,
            drop_last=self.args.dataloader_drop_last,
            process_rank=process_rank,
            num_processes=num_processes,
            generator=None,
            seed=seed,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=True if getattr(self.args, "dataloader_num_workers", 0) > 0 else False,
            prefetch_factor=4 if getattr(self.args, "dataloader_num_workers", 0) > 0 else None,
        )


STATIC_SFT_CLI_DEFAULTS = {
    "comp_ratio_or_len": 0.25,
    "encoder_base_model_path": "/share/models/Qwen3.5-0.8B",
    "decoder_base_model_path": "/share/models/Qwen3.5-0.8B",
    "mlp_converter_hidden_dim": 4096,
    "placeholder_id": 248076,
    "memory_token_begin_id": 247000,
    "feature_extract_method": "mean_pooling",
    "encoder_training": "lora",
    "decoder_training": "lora",
    "context_max_length": 1034,
    "context_min_length": 64,
    "decoder_max_length": 1024,
    "decoder_min_length": 1,
    "dataset_num_proc": 40,
    "dataset_exclude_sign": "eval",
    "template_dir": ".",
    "add_eos_token_to_context": True,
    "use_gradient_checkpointing": False,
    "per_device_train_batch_size": 10,
    "max_steps": 40000,
    "gradient_accumulation_steps": 1,
    "learning_rate": 4e-5,
    "warmup_steps": 100,
    "eval_steps": 500,
    "save_steps": 10000,
    "save_total_limit": 3,
    "logging_steps": 20,
    "lr_scheduler_type": "constant_with_warmup",
    "seed": 0,
    "group_by_length": True,
    "num_buckets": 10,
    "collator_pad_to_multiple_of": None,
}


def main():
    parser = argparse.ArgumentParser(description="Train CtxCompModel (static)")
    add_ctxcomp_sft_cli_args(parser, STATIC_SFT_CLI_DEFAULTS)
    args = parser.parse_args()

    if torch.distributed.is_initialized():
        try:
            store = torch.distributed.get_store()
            if store is not None and hasattr(store, "set_timeout"):
                store.set_timeout(4 * 3600)
        except Exception:
            pass

    output_dir = (
        f"./training_output/ctxcomp-static-{args.feature_extract_method}-contextlen={args.context_max_length}to{args.context_min_length}-"
        f"comp={format_comp_ratio_for_output_dir(args.comp_ratio_or_len)}-enc={pathlib.Path(args.encoder_base_model_path).name}-{args.encoder_training}-dec={pathlib.Path(args.decoder_base_model_path).name}-{args.decoder_training}"
    )
    train_args = build_ctxcomp_training_arguments(
        args,
        output_dir=output_dir,
        ddp_find_unused_parameters=not args.use_gradient_checkpointing,
        gradient_checkpointing=False,
    )

    data_config = DEFAULT_DATA_CONFIG

    # data_config = [{"dir": "...", "task": "qa_and_sum", "max_files": 10, "max_samples": 10000}]

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

    #gradient_checkpointing_enable
    if args.use_gradient_checkpointing:
        encoder.gradient_checkpointing_enable({"use_reentrant": False})
        decoder.gradient_checkpointing_enable({"use_reentrant": False})

    model = CtxCompModel(
        encoder=encoder,
        decoder=decoder,
        placeholder_token_id=args.placeholder_id,
        memory_token_begin_id=args.memory_token_begin_id,
        comp_ratio_or_len=args.comp_ratio_or_len,
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

    with train_args.main_process_first(desc="Loading dataset"):
        ds = load_from_data_config(
            data_config=data_config,
            tokenizer_decoder=tokenizer_decoder,
            tokenizer_encoder=tokenizer_encoder,
            num_proc=args.dataset_num_proc,
            max_length=args.decoder_max_length,
            min_length=args.decoder_min_length,
            context_max_length=args.context_max_length,
            context_min_length=args.context_min_length,
            comp_ratio_or_len=args.comp_ratio_or_len,
            placeholder_token=placeholder_token,
            feature_extract_method=args.feature_extract_method,
            seed=train_args.seed,
            exclude_sign=args.dataset_exclude_sign or None,
            add_eos_token_to_context=args.add_eos_token_to_context,
            generation_path=args.decoder_base_model_path,
            template_dir=args.template_dir,
        )
        ds = ds.shuffle(seed=train_args.seed)
        print("Dataset size:", len(ds))

    _collator_kw = {}
    if args.collator_pad_to_multiple_of is not None:
        _collator_kw["pad_to_multiple_of"] = args.collator_pad_to_multiple_of
    collator = CtxCompDataCollator(
        tokenizer=tokenizer_decoder,
        tokenizer_encoder=tokenizer_encoder,
        padding=True,
        max_length=args.context_max_length,
        label_pad_token_id=-100,
        **_collator_kw,
    )
    train_ds = ds.shuffle(seed=train_args.seed)

    if args.group_by_length:
        trainer = BucketedCtxCompTrainer(
            model=model,
            args=train_args,
            data_collator=collator,
            train_dataset=train_ds,
            eval_dataset=None,
            processing_class=tokenizer_decoder,
            length_column_name="context_len",
            num_buckets=args.num_buckets,
        )
    else:
        trainer = CtxCompTrainer(
            model=model,
            args=train_args,
            data_collator=collator,
            train_dataset=train_ds,
            eval_dataset=None,
            processing_class=tokenizer_decoder,
        )

    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup accelerate launch --config_file ddp_config.yaml --num_processes 8 sft_ctxcomp_static.py >sft_qwen3.5.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0  accelerate launch --config_file ddp_config.yaml --num_processes 1 sft_ctxcomp_static.py
