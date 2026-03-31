"""
SFT script for CtxCompSemiDynamicModel: multi-ratio with predicted compression ratio.

Uses single placeholder per sample; model predicts M and expands. Train with compress_len_labels
(log2(context_length/summary_length)). Collator sets comp_ratio_or_len_override per batch and passes compress_len_labels.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import pathlib
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Model,AutoModelForCausalLM,AutoModel
from transformers.trainer_utils import is_main_process
from typing import List, Dict, Any, Optional

import sft_ctxcomp_static as ctxcomp_sft
import sft_ctxcomp_static_multiratio as multiratio_sft

from modeling_ctxcomp import CtxCompSemiDynamicModel
from sft_ctxcomp_train_config import (
    add_ctxcomp_sft_cli_args,
    build_ctxcomp_training_arguments,
    comp_ratio_or_len_as_tuple,
    format_comp_ratio_for_output_dir,
)
from dataset_utils_ctxcomp import DEFAULT_DATA_CONFIG,TEST_DATA_CONFIG

os.environ.setdefault("WANDB_PROJECT", "SFT_ctxcomp_semi_dynamic")

DEFAULT_RATIO_WEIGHT = None #[1, 1.2, 1.2, 1.4, 1.4]

SEMI_DYNAMIC_SFT_CLI_DEFAULTS = {
    "comp_ratio_or_len": (0.25, 0.125, 0.0625),
    "encoder_base_model_path": "/share/models/Qwen3.5-0.8B",
    "decoder_base_model_path": "/share/models/Qwen3.5-0.8B",
    "mlp_converter_hidden_dim": 4096,
    "placeholder_id": 248076,
    "memory_token_begin_id": 247000,
    "feature_extract_method": "mean_pooling",
    "encoder_training": "lora",
    "decoder_training": "lora",


    "context_max_length": 1300,
    "context_min_length": 64,
    "decoder_max_length": 1024,
    "decoder_min_length": 1,
    "dataset_num_proc": 40,
    "dataset_exclude_sign": "eval",
    "template_dir": ".",
    "add_eos_token_to_context": True,


    "use_gradient_checkpointing": False,
    "per_device_train_batch_size": 10,
    "max_steps": 100000,
    "gradient_accumulation_steps": 1,
    "learning_rate": 4e-5,
    "warmup_steps": 100,
    "eval_steps": 500,
    "save_steps": 20000,
    "save_total_limit": 5,
    "logging_steps": 20,
    "lr_scheduler_type": "constant_with_warmup",
    "seed": 0,
    "group_by_length": True,
    "num_buckets": 10,
    "collator_pad_to_multiple_of": None,
}

load_single_ratio_dataset = multiratio_sft.load_single_ratio_dataset


class CtxCompLoggingMixin:
    """Log lm_loss and comp_ratio_loss to wandb for CtxCompSemiDynamicModel."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        loss, outputs = result
        lm = getattr(outputs, "lm_loss", None) if not isinstance(outputs, dict) else outputs.get("lm_loss")
        cr = getattr(outputs, "comp_ratio_loss", None) if not isinstance(outputs, dict) else outputs.get("comp_ratio_loss")
        if cr is None:
            raise ValueError("Batch must provide comp_ratio_loss (compress_len_labels in data).")
        if lm is None:
            raise ValueError("Model must return lm_loss.")
        device = lm.device
        if not hasattr(self, "_ctxcomp_lm_sum"):
            self._ctxcomp_lm_sum = torch.zeros_like(lm, device=device)
        self._ctxcomp_lm_sum = self._ctxcomp_lm_sum + lm.detach()
        if not hasattr(self, "_ctxcomp_cr_sum"):
            self._ctxcomp_cr_sum = torch.zeros_like(cr, device=device)
        self._ctxcomp_cr_sum = self._ctxcomp_cr_sum + cr.detach()
        self._ctxcomp_n_steps = getattr(self, "_ctxcomp_n_steps", 0) + 1
        return (loss, outputs) if return_outputs else loss

    def _ctxcomp_gather_mean(self, t: torch.Tensor) -> float:
        try:
            from transformers.trainer_pt_utils import nested_gather
            if getattr(self.args, "parallel_mode", None) is not None:
                return nested_gather(t, self.args.parallel_mode).mean().item()
        except Exception:
            pass
        if t.numel() == 1:
            return t.item()
        return t.mean().item()

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs):
        n = getattr(self, "_ctxcomp_n_steps", 0)
        if n > 0 and hasattr(self, "_ctxcomp_lm_sum"):
            try:
                logs["lm_loss"] = round(self._ctxcomp_gather_mean(self._ctxcomp_lm_sum) / n, 6)
            except Exception:
                pass
            self._ctxcomp_lm_sum = torch.zeros_like(self._ctxcomp_lm_sum, device=self._ctxcomp_lm_sum.device)
        if n > 0 and hasattr(self, "_ctxcomp_cr_sum"):
            try:
                logs["comp_ratio_loss"] = round(self._ctxcomp_gather_mean(self._ctxcomp_cr_sum) / n, 6)
            except Exception:
                pass
            self._ctxcomp_cr_sum = torch.zeros_like(self._ctxcomp_cr_sum, device=self._ctxcomp_cr_sum.device)
        if n > 0:
            self._ctxcomp_n_steps = 0
        super().log(logs, start_time=start_time, **kwargs)


class DynamicCtxCompTrainer(CtxCompLoggingMixin, ctxcomp_sft.CtxCompTrainer):
    """CtxCompTrainer + logging of lm_loss and comp_ratio_loss."""
    pass


class DynamicBucketedCtxCompTrainer(CtxCompLoggingMixin, ctxcomp_sft.BucketedCtxCompTrainer):
    """BucketedCtxCompTrainer + logging of lm_loss and comp_ratio_loss."""
    pass


class MultiratioDynamicCtxCompCollator(multiratio_sft.MultiratioCtxCompCollator):
    """Add compress_len_labels to batch for CtxCompSemiDynamicModel."""

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        batch = super().__call__(features, return_tensors=return_tensors)
        if features and "compress_len_labels" in features[0]:
            batch["compress_len_labels"] = torch.tensor(
                [f["compress_len_labels"] for f in features], dtype=torch.float32
            )
        return batch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train CtxCompSemiDynamicModel")
    add_ctxcomp_sft_cli_args(parser, SEMI_DYNAMIC_SFT_CLI_DEFAULTS)
    args = parser.parse_args()

    if torch.distributed.is_initialized():
        store = torch.distributed.get_store()
        if store is not None and hasattr(store, "set_timeout"):
            store.set_timeout(4 * 3600)

    comp_ratio_options = comp_ratio_or_len_as_tuple(args.comp_ratio_or_len)

    output_dir = (
        f"./training_output/ctxcomp-semi-dynamic-{args.feature_extract_method}-contextlen={args.context_max_length}to{args.context_min_length}-"
        f"ratios={format_comp_ratio_for_output_dir(args.comp_ratio_or_len)}-enc={pathlib.Path(args.encoder_base_model_path).name}-{args.encoder_training}-dec={pathlib.Path(args.decoder_base_model_path).name}-{args.decoder_training}"
    )
    train_args = build_ctxcomp_training_arguments(
        args,
        output_dir=output_dir,
        ddp_find_unused_parameters=not args.use_gradient_checkpointing,
        gradient_checkpointing=False,
    )

    data_config = DEFAULT_DATA_CONFIG#TEST_DATA_CONFIG #

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

    model = CtxCompSemiDynamicModel(
        encoder=encoder,
        decoder=decoder,
        placeholder_token_id=args.placeholder_id,
        memory_token_begin_id=args.memory_token_begin_id,
        comp_ratio_or_len=comp_ratio_options,
        mlp_converter_hidden_dim=args.mlp_converter_hidden_dim,
        feature_extract_method=args.feature_extract_method,
        encoder_training=args.encoder_training,
        decoder_training=args.decoder_training,
        discretize_ratio_mode="round",
        discretize_compare_in_log=False,
    )

    from peft import LoraConfig
    emb_lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.1, bias="none", use_rslora=True)
    gen_lora_config = LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.1, bias="none", use_rslora=True, task_type="CAUSAL_LM")
    model.set_trainable_params(emb_lora_config, gen_lora_config)
    model.reinit_memory_tokens()
    model.print_trainable_parameters()

    with train_args.main_process_first(desc="Loading dataset with compress_len_labels"):
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
            add_compress_len_labels=True,
            add_eos_token_to_context=args.add_eos_token_to_context,
            generation_path=args.decoder_base_model_path,
            template_dir=args.template_dir,
        )
        print("Dataset size (with compress_len_labels):", len(ds))

    _collator_kw = {}
    if args.collator_pad_to_multiple_of is not None:
        _collator_kw["pad_to_multiple_of"] = args.collator_pad_to_multiple_of
    data_collator = MultiratioDynamicCtxCompCollator(
        comp_ratio_or_len_options=list(comp_ratio_options),
        placeholder_id=args.placeholder_id,
        feature_extract_method=args.feature_extract_method,
        ratios_weight=DEFAULT_RATIO_WEIGHT,
        tokenizer=tokenizer_decoder,
        tokenizer_encoder=tokenizer_encoder,
        padding=True,
        max_length=args.context_max_length,
        label_pad_token_id=-100,
        **_collator_kw,
    )

    if args.group_by_length:
        trainer = DynamicBucketedCtxCompTrainer(
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
        trainer = DynamicCtxCompTrainer(
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

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup accelerate launch --config_file ddp_config.yaml --num_processes 8 sft_ctxcomp_semi_dynamic.py >sft_semi_dynamic.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0  accelerate launch --config_file ddp_config.yaml --num_processes 1 sft_ctxcomp_semi_dynamic.py