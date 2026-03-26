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

import torch
from transformers import TrainingArguments, AutoTokenizer, Qwen3ForCausalLM, Qwen3Model
from transformers.trainer_utils import is_main_process
from typing import List, Dict, Any, Optional, Union

import sft_ctxcomp_static as ctxcomp_sft
import sft_ctxcomp_static_multiratio as multiratio_sft

from modeling_ctxcomp import CtxCompSemiDynamicModel

os.environ.setdefault("WANDB_PROJECT", "SFT_ctxcomp_semi_dynamic")

DEFAULT_COMP_RATIO_OR_LEN_OPTIONS: List[Union[float, int]] = [0.5, 0.25, 0.125, 0.0625, 0.03125]
DEFAULT_RATIO_WEIGHT = [1, 1.2, 1.2, 1.4, 1.4]

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
    parser.add_argument("--doc_max_length", type=int, default=1200)
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

    max_length = 1200
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
        f"./training_output/ctxcomp-semi-dynamic-{feature_extract_method}-doclen={doc_max_length}to{doc_min_length}-"
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        logging_steps=20,
        bf16=True,
        fp16=False,
        seed=0,
        save_total_limit=5,
        torch_compile=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        dataloader_drop_last=True,
    )

    data_config = [
        {"dir": "../数据合成/doc_sum_qa_shortsum_synthetic_text_128_en/Qwen3-30B-A3B-Instruct-2507__vllm", "task": "qa_and_sum", "max_files": 50, "max_samples": 1000000},
    ]

    embedding_path = "/share/models/Qwen3-Embedding-0.6B"
    generation_path = "/share/models/Qwen3-4B-Instruct-2507"

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

    model = CtxCompSemiDynamicModel(
        encoder=encoder,
        decoder=decoder,
        placeholder_token_id=placeholder_id,
        memory_token_begin_id=memory_token_begin_id,
        comp_ratio_or_len=comp_ratio_or_len_init,
        mlp_converter_hidden_dim=4096,
        feature_extract_method=feature_extract_method,
        encoder_training=encoder_training,
        decoder_training=decoder_training,
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
            add_compress_len_labels=True,
        )
        print("Dataset size (with compress_len_labels):", len(ds))

    data_collator = MultiratioDynamicCtxCompCollator(
        comp_ratio_or_len_options=DEFAULT_COMP_RATIO_OR_LEN_OPTIONS,
        placeholder_id=placeholder_id,
        feature_extract_method=feature_extract_method,
        ratios_weight=DEFAULT_RATIO_WEIGHT,
        tokenizer=tokenizer_decoder,
        tokenizer_encoder=tokenizer_encoder,
        padding=True,
        max_length=doc_max_length,
        label_pad_token_id=-100,
    )

    if GROUP_BY_LENGTH:
        trainer = DynamicBucketedCtxCompTrainer(
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
