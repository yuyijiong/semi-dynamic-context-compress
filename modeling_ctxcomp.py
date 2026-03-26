"""
Context compression models: CtxCompModel and CtxCompSemiDynamicModel.

Encoder compresses a long document into a fixed or variable number of vectors;
a projector (MLP) maps them to the decoder's hidden space; the decoder generates
with these vectors injected at placeholder token positions in the query/prompt.

See CtxCompModel and CtxCompSemiDynamicModel docstrings for parameter meanings and usage.
"""

import os
import math
import json
import pathlib
import warnings
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM,PreTrainedModel,Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.activations import GELUActivation
from peft import PeftModel, LoraConfig, get_peft_model

# Feature extraction methods that interpret comp_ratio_or_len as a ratio in (0, 1].
# For these, compression length M is derived from document length L (e.g. M = ceil(L * ratio) or pool_size = 1/ratio).
RATIO_BASED_FEATURE_EXTRACT_METHODS = ("mean_pooling", "mean_pooling_causal")

# All allowed values for feature_extract_method. Others (e.g. last_tokens, same_memory_tokens)
# use comp_ratio_or_len as a positive integer (number of compressed tokens M).
SUPPORTED_FEATURE_EXTRACT_METHODS = (
    "mean_pooling",
    "mean_pooling_causal",
    "last_tokens",
    "same_memory_tokens",
    "different_memory_tokens",
)


def _validate_and_normalize_comp_ratio_or_len(
    comp_ratio_or_len: Union[int, float, Tuple[Union[int, float], ...], List[Union[int, float]]],
    feature_extract_method: str,
) -> Union[int, float, Tuple[Union[int, float], ...]]:
    """
    Validate and normalize comp_ratio_or_len: scalar or tuple/list for multiple options.

    Args:
        comp_ratio_or_len: Either a single value or a non-empty tuple/list of values.
            - For ratio-based methods (mean_pooling, mean_pooling_causal): each value must be in (0, 1].
            - For other methods: each value must be a positive integer (number of compressed tokens).
        feature_extract_method: One of SUPPORTED_FEATURE_EXTRACT_METHODS; determines how comp_ratio_or_len is interpreted.

    Returns:
        Normalized scalar (int or float) or tuple of values (list is converted to tuple).
    """
    if feature_extract_method not in SUPPORTED_FEATURE_EXTRACT_METHODS:
        raise ValueError(
            f"feature_extract_method must be one of {SUPPORTED_FEATURE_EXTRACT_METHODS}, got {feature_extract_method!r}"
        )
    ratio_based = feature_extract_method in RATIO_BASED_FEATURE_EXTRACT_METHODS
    if isinstance(comp_ratio_or_len, (list, tuple)):
        comp_ratio_or_len = tuple(comp_ratio_or_len)
        if not comp_ratio_or_len:
            raise ValueError("comp_ratio_or_len as tuple/list must be non-empty")
        if ratio_based:
            for v in comp_ratio_or_len:
                if not isinstance(v, (int, float)) or not (0 < v <= 1):
                    raise ValueError(
                        f"When feature_extract_method is '{feature_extract_method}', each element of "
                        f"comp_ratio_or_len must be a float in (0, 1]"
                    )
            return tuple(float(v) for v in comp_ratio_or_len)
        for v in comp_ratio_or_len:
            if not isinstance(v, int) or v < 1:
                raise ValueError(
                    "When feature_extract_method is not ratio-based, each element of comp_ratio_or_len must be a positive int"
                )
        return comp_ratio_or_len
    if ratio_based:
        if not isinstance(comp_ratio_or_len, (int, float)) or not (0 < comp_ratio_or_len <= 1):
            raise ValueError(
                f"When feature_extract_method is '{feature_extract_method}', comp_ratio_or_len must be a float in (0, 1]"
            )
        return float(comp_ratio_or_len) if isinstance(comp_ratio_or_len, int) else comp_ratio_or_len
    if not isinstance(comp_ratio_or_len, int) or comp_ratio_or_len < 1:
        raise ValueError(
            "When feature_extract_method is not ratio-based, comp_ratio_or_len must be a positive int"
        )
    return comp_ratio_or_len


def _comp_ratio_or_len_to_capacity(
    comp_ratio_or_len: Union[int, float, Tuple, List],
) -> int:
    """
    Convert comp_ratio_or_len to a single integer capacity.
    If tuple/list: returns max (e.g. max number of memory tokens). If scalar: returns int(comp_ratio_or_len).
    Used for sequence length checks and memory token index range.
    """
    if isinstance(comp_ratio_or_len, (tuple, list)):
        return int(max(comp_ratio_or_len))
    return int(comp_ratio_or_len)


def _comp_ratio_or_len_for_max_memory(
    comp_ratio_or_len: Union[int, float, Tuple, List],
) -> Union[int, float]:
    """
    Scalar used when preparing encoder input for the "worst case" (most memory tokens).
    If comp_ratio_or_len is ratio-based (float): returns min(ratios). If int-based: returns max(ints).
    """
    if isinstance(comp_ratio_or_len, (tuple, list)):
        if not comp_ratio_or_len:
            raise ValueError("comp_ratio_or_len as tuple/list must be non-empty")
        if isinstance(comp_ratio_or_len[0], float) or any(isinstance(v, float) for v in comp_ratio_or_len):
            return min(float(v) for v in comp_ratio_or_len)
        return max(int(v) for v in comp_ratio_or_len)
    return comp_ratio_or_len if isinstance(comp_ratio_or_len, float) else int(comp_ratio_or_len)


def _validate_forward_comp_ratio_or_len(
    model_comp_ratio_or_len: Union[int, float, Tuple, List],
    passed_comp_ratio_or_len: Optional[Union[int, float]],
) -> Optional[Union[int, float]]:
    """
    Validate comp_ratio_or_len passed to forward() or generate().

    Args:
        model_comp_ratio_or_len: The model's comp_ratio_or_len (may be a tuple for multi-ratio models).
        passed_comp_ratio_or_len: Value passed by the caller (e.g. comp_ratio_or_len_override).

    Returns:
        The scalar to use for this call, or None to keep the model's default (when model has a single value).
    Raises:
        ValueError: If model has multiple ratios but caller did not pass one, or passed value is not in the allowed list.
    """
    if isinstance(model_comp_ratio_or_len, (tuple, list)):
        if passed_comp_ratio_or_len is None:
            raise ValueError(
                "comp_ratio_or_len must be specified when the model supports multiple ratios; "
                f"allowed values: {model_comp_ratio_or_len}"
            )
        if passed_comp_ratio_or_len not in model_comp_ratio_or_len:
            raise ValueError(
                f"comp_ratio_or_len={passed_comp_ratio_or_len} must be one of {model_comp_ratio_or_len}"
            )
        return passed_comp_ratio_or_len
    return passed_comp_ratio_or_len


def _comp_ratio_or_len_to_serializable(
    comp_ratio_or_len: Union[int, float, Tuple, List],
) -> Union[int, float, List]:
    """For config save: tuple -> list for JSON."""
    if isinstance(comp_ratio_or_len, (tuple, list)):
        return list(comp_ratio_or_len)
    return comp_ratio_or_len


def _last_content_indices_from_attention_mask(
    doc_attention_mask: Optional[torch.Tensor],
    L_per_sample: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Index of last 1 in attention_mask per sample (left-padding safe).
    If doc_attention_mask is None, returns L_per_sample - 1.
    Returns [batch_size] long tensor.
    """
    if doc_attention_mask is not None:
        seq_len = doc_attention_mask.shape[1]
        return (
            doc_attention_mask.long()
            * torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        ).max(dim=1)[1]
    return L_per_sample - 1


def _get_embed_tokens_module(model):
    """Return embed_tokens from base model (works for AutoModel and AutoModelForCausalLM, with or without PEFT)."""
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
        return base.model.embed_tokens
    return getattr(base, "embed_tokens", None)


def _hidden_size_from_config(config) -> int:
    """Top-level hidden_size, or text_config.hidden_size (e.g. Qwen3.5 vs Qwen3)."""
    hs = getattr(config, "hidden_size", None)
    if hs is not None:
        return hs
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        hs = getattr(text_cfg, "hidden_size", None)
        if hs is not None:
            return hs
    raise AttributeError(
        f"{type(config).__name__!r} has no 'hidden_size' at top level or under 'text_config'"
    )


def _model_hidden_size(module) -> int:
    """Backbone hidden size for encoder/decoder (PEFT unwrap + nested text config)."""
    base = module.get_base_model() if isinstance(module, PeftModel) else module
    return _hidden_size_from_config(base.config)


class CtxCompConverterMLP(nn.Module):
    """
    MLP that maps encoder hidden size to decoder hidden size.

    Architecture: RMSNorm -> (optional gate branch) -> up_proj -> activation -> down_proj.
    When gate=True, uses gate_proj and gated multiplication (similar to Qwen MLP).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        intermediate_size: int,
        gate: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Args:
            input_dim: Encoder hidden size (input dimension).
            output_dim: Decoder hidden size (output dimension). Can differ from encoder.
            intermediate_size: Hidden size of the middle layer (MLP bottleneck). Larger values
                increase capacity but parameters. Typical range 2048--8192.
            gate: If True, use a gated structure (gate_proj * up_proj) before down_proj; if False, single up_proj.
            device: Device for parameters (e.g. decoder.device).
            dtype: Data type for parameters (e.g. decoder.dtype).
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intermediate_size = intermediate_size
        self.gate = gate
        if self.gate:
            self.gate_proj = nn.Linear(self.input_dim, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(self.input_dim, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_dim, bias=False, device=device, dtype=dtype)
        self.act_fn = GELUActivation()
        self.rms_norm = Qwen3RMSNorm(self.input_dim, eps=1e-6).to(device).to(dtype)

    def forward(self, x):
        if self.gate:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(self.rms_norm(x)))
        return self.down_proj(self.act_fn(self.up_proj(self.rms_norm(x))))


class CtxCompModel(nn.Module):
    """
    Context compression model with fixed compression length or ratio.

    Flow: document -> encoder -> extract features (by feature_extract_method) -> projector -> decoder.
    The decoder's input sequence contains placeholder token(s); those positions are replaced by
    the projected document features. You can train encoder/decoder (full or LoRA) and the projector.

    Supported feature_extract_method and comp_ratio_or_len semantics:
    - "last_tokens": comp_ratio_or_len = int M; use last M token hidden states.
    - "same_memory_tokens": comp_ratio_or_len = int M; append M identical special tokens, use their hiddens.
    - "different_memory_tokens": comp_ratio_or_len = int M; append M distinct special tokens, use their hiddens.
    - "mean_pooling": comp_ratio_or_len = float in (0,1] (ratio); pool_size = 1/ratio, bidirectional encoder.
    - "mean_pooling_causal": same ratio semantics, causal encoder.
    """

    def __init__(
        self,
        encoder,
        decoder,
        placeholder_token_id: int,
        memory_token_begin_id: int = 151000,
        comp_ratio_or_len: Union[int, float, Tuple[Union[int, float], ...], List[Union[int, float]]] = 1,
        mlp_converter_hidden_dim: int = 4096,
        feature_extract_method: str = "mean_pooling",
        encoder_training: str = "none",
        decoder_training: str = "none",
        encoder_each_layer_sliding_window: Optional[List[int]] = None,
    ):
        """
        Args:
            encoder: Backbone for encoding the document (e.g. AutoModel or PeftModel).
                Output last_hidden_state is used to extract compressed features.
            decoder: Causal LM for generation (e.g. AutoModelForCausalLM or PeftModel).
                Receives inputs_embeds where placeholder positions are replaced by projected doc features.
            placeholder_token_id: Token id used in the decoder input to mark where compressed context
                should be injected. The number of placeholders per sample must match the number of
                compressed tokens M for that sample (or comp_ratio_or_len when fixed).
            memory_token_begin_id: For "same_memory_tokens" / "different_memory_tokens", the first token id
                of the special memory tokens in the vocabulary. Must be within the encoder's vocab.
                Default 151000 is a common extension range for Qwen.
            comp_ratio_or_len: Compression length or ratio. Interpretation depends on feature_extract_method:
                - For "last_tokens", "same_memory_tokens", "different_memory_tokens": positive int (number of tokens M).
                  Can be a tuple/list of ints to support multiple fixed options; then forward/generate must receive
                  comp_ratio_or_len_override to choose one.
                - For "mean_pooling", "mean_pooling_causal": float in (0, 1] (compression ratio). Pool size is
                  round(1/ratio); M = ceil(L / pool_size). Can be tuple/list of floats for multiple ratios.
            mlp_converter_hidden_dim: Hidden size of the projector MLP (intermediate layer). Larger values
                increase capacity; typical 2048--8192. Only the projector is this dimension; input/output
                dims are taken from encoder and decoder config.
            feature_extract_method: How to obtain M vectors from encoder output. One of:
                "mean_pooling", "mean_pooling_causal", "last_tokens", "same_memory_tokens", "different_memory_tokens".
                See class docstring for semantics.
            encoder_training: How to train the encoder: "full" (all params), "lora", or "none" (frozen).
            decoder_training: How to train the decoder: "full", "lora", or "none".
            encoder_each_layer_sliding_window: Optional list of sliding window sizes, one per encoder layer.
                Length must equal encoder number of layers. Used to limit attention span per layer (e.g. for long docs).
                None means do not modify encoder attention.
        """
        super().__init__()
        self.comp_ratio_or_len = _validate_and_normalize_comp_ratio_or_len(comp_ratio_or_len, feature_extract_method)
        self.placeholder_token_id = placeholder_token_id
        self.feature_extract_method = feature_extract_method
        self.memory_token_begin_id = memory_token_begin_id
        self.encoder_training = encoder_training
        self.decoder_training = decoder_training
        self.encoder_each_layer_sliding_window = encoder_each_layer_sliding_window
        self.encoder = encoder
        self.decoder = decoder
        self._shared_base_model = False

        if self.feature_extract_method == "mean_pooling":
            self._set_encoder_attention_non_causal()

        if self.encoder_each_layer_sliding_window is not None:
            self._set_encoder_sliding_window()

        self._memory_token_indices = self._get_memory_token_indices()

        embed_dim = _model_hidden_size(self.encoder)
        gen_dim = _model_hidden_size(self.decoder)
        self.projector = CtxCompConverterMLP(
            input_dim=embed_dim,
            output_dim=gen_dim,
            intermediate_size=mlp_converter_hidden_dim,
            device=self.decoder.device,
            dtype=self.decoder.dtype,
        )

    def _set_encoder_attention_non_causal(self):
        """Set is_causal=False for all attention modules in encoder (for bidirectional mean_pooling)."""
        base = self.encoder.get_base_model() if isinstance(self.encoder, PeftModel) else self.encoder
        for module in base.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
                warnings.warn(f"Set module {module.__class__.__name__} is_causal to False")

    def _set_encoder_sliding_window(self):
        """Set per-layer sliding_window for encoder attention."""
        base = self.encoder.get_base_model() if isinstance(self.encoder, PeftModel) else self.encoder
        layers = getattr(base, "layers", None) or getattr(getattr(base, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Encoder has no model.layers; cannot set per-layer sliding_window.")
        num_layers = len(layers)
        if len(self.encoder_each_layer_sliding_window) != num_layers:
            raise ValueError(
                f"encoder_each_layer_sliding_window length ({len(self.encoder_each_layer_sliding_window)}) "
                f"must equal encoder number of layers ({num_layers})."
            )
        for i, (layer, window_size) in enumerate(zip(layers, self.encoder_each_layer_sliding_window)):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise ValueError(f"Encoder layer {i} has no self_attn.")
            setattr(attn, "sliding_window", window_size)

    def _get_memory_token_indices(self):
        """Token indices for memory tokens (for training only those embedding rows)."""
        if self.feature_extract_method == "same_memory_tokens":
            return [self.memory_token_begin_id]
        if self.feature_extract_method == "different_memory_tokens":
            n = _comp_ratio_or_len_to_capacity(self.comp_ratio_or_len)
            return list(range(self.memory_token_begin_id, self.memory_token_begin_id + n))
        return []

    def _set_memory_token_embeddings_trainable(self, model):
        """Make only memory token embedding rows trainable; zero gradient for others via hook."""
        if not self._memory_token_indices:
            return
        embed_tokens = _get_embed_tokens_module(model)
        if embed_tokens is None:
            return
        embed_tokens.weight.requires_grad = True
        if hasattr(embed_tokens, "_memory_embed_hook_handle"):
            return
        indices = self._memory_token_indices

        def _grad_hook(grad):
            if grad is None:
                return grad
            mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
            for i in indices:
                mask[i] = True
            out = grad.clone()
            out[~mask] = 0
            return out

        embed_tokens._memory_embed_hook_handle = embed_tokens.weight.register_hook(_grad_hook)

    def _switch_encoder_adapter(self):
        """Switch to encoder adapter when using shared base model."""
        if self._shared_base_model and isinstance(self.encoder, PeftModel) and "encoder" in self.encoder.peft_config:
            self.encoder.set_adapter("encoder")

    def _switch_decoder_adapter(self):
        """Switch to decoder adapter when using shared base model."""
        if self._shared_base_model and isinstance(self.decoder, PeftModel) and "decoder" in self.decoder.peft_config:
            self.decoder.set_adapter("decoder")

    def replace_placeholder_tokens(
        self,
        input_ids: torch.Tensor,
        compressed_doc_features: torch.Tensor,
        doc_valid_mask: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        doc_input_ids: Optional[torch.Tensor] = None,
        comp_ratio_or_len_override: Optional[Union[int, float]] = None,
    ):
        """
        Replace placeholder positions in input_ids with compressed_doc_features.

        Args:
            input_ids: [batch_size, seq_len] decoder input containing placeholder_token_id.
            compressed_doc_features: [batch_size, M_or_max, hidden_dim] projected document features.
            doc_valid_mask: [batch_size, M_or_max] True where the feature is valid (per-sample M may differ).
                For each sample, the number of True must equal the number of placeholder_token_id in input_ids.
            doc_attention_mask: Optional; used only for error messages (context length).
            doc_input_ids: Optional; used only for error messages (max doc length).
            comp_ratio_or_len_override: Optional; used only for error messages.

        Returns:
            inputs_embeds: [batch_size, seq_len, hidden_dim] with placeholders replaced by compressed_doc_features.
        """
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        placeholder_mask = (input_ids == self.placeholder_token_id)
        n_placeholder_tokens = placeholder_mask.sum(dim=1)
        num_valid_tokens = doc_valid_mask.sum(dim=1)
        hidden_dim = compressed_doc_features.shape[-1]

        for i in range(input_ids.shape[0]):
            if n_placeholder_tokens[i].item() != num_valid_tokens[i].item():
                used = comp_ratio_or_len_override if comp_ratio_or_len_override is not None else getattr(self, "comp_ratio_or_len", None)
                ctx_valid_len = (
                    doc_attention_mask[i].sum().item()
                    if doc_attention_mask is not None and i < doc_attention_mask.shape[0]
                    else None
                )
                batch_max_doc_len = doc_input_ids.shape[1] if doc_input_ids is not None else None
                extra = [
                    f"comp_ratio_or_len={used}",
                    f"sample_{i}_context_valid_len={ctx_valid_len}" if ctx_valid_len is not None else "sample_{i}_context_valid_len=N/A",
                    f"batch_max_doc_len={batch_max_doc_len}" if batch_max_doc_len is not None else "batch_max_doc_len=N/A",
                ]
                raise ValueError(
                    f"Sample {i}: placeholder count {n_placeholder_tokens[i].item()} must equal "
                    f"doc valid count {num_valid_tokens[i].item()} (id={self.placeholder_token_id}). Debug: {', '.join(extra)}"
                )

        compressed_flat = compressed_doc_features[doc_valid_mask].reshape(-1)
        placeholder_mask_expanded = placeholder_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(placeholder_mask_expanded, compressed_flat)
        return inputs_embeds

    def _prepare_doc_encoder_inputs(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        comp_ratio_or_len_arg: Optional[Union[int, float, Tuple, List]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Prepare encoder input; append memory tokens for memory_tokens methods.
        Returns (model_input_ids, model_attention_mask, L_per_sample [B]).
        """
        seq_len = doc_input_ids.shape[1]
        if doc_attention_mask is not None:
            L_per_sample = doc_attention_mask.sum(dim=1).clamp(min=1).long()
        else:
            L_per_sample = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        is_memory = self.feature_extract_method in ("same_memory_tokens", "different_memory_tokens")
        if not is_memory:
            return doc_input_ids, doc_attention_mask, L_per_sample

        effective = comp_ratio_or_len_arg if comp_ratio_or_len_arg is not None else self.comp_ratio_or_len
        max_mem = int(effective) if not isinstance(effective, (tuple, list)) else max(int(x) for x in effective)

        with torch.no_grad():
            if self.feature_extract_method == "same_memory_tokens":
                memory_tokens = torch.full(
                    (batch_size, max_mem), self.memory_token_begin_id,
                    dtype=doc_input_ids.dtype, device=device,
                )
            else:
                memory_tokens = torch.tensor(
                    list(range(self.memory_token_begin_id, self.memory_token_begin_id + max_mem)),
                    dtype=doc_input_ids.dtype, device=device,
                ).unsqueeze(0).repeat(batch_size, 1)
            model_input_ids = torch.cat([doc_input_ids, memory_tokens], dim=1)
            if doc_attention_mask is not None:
                memory_mask = torch.ones((batch_size, max_mem), dtype=doc_attention_mask.dtype, device=device)
                model_attention_mask = torch.cat([doc_attention_mask, memory_mask], dim=1)
            else:
                model_attention_mask = torch.ones_like(model_input_ids)
        return model_input_ids, model_attention_mask, L_per_sample

    def _get_doc_encoder_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run encoder and return last_hidden_state. Handles shared base model (adapter switch + CausalLM backbone)."""
        self._switch_encoder_adapter()
        model = self.encoder
        if self._shared_base_model and hasattr(model, "model"):
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask,use_cache=False)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,use_cache=False)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        raise AttributeError("Encoder output has no last_hidden_state or hidden_states")

    def _extract_doc_features_from_last_hidden_last_tokens(
        self,
        last_hidden: torch.Tensor,
        L_max: int,
        L_per_sample: torch.Tensor,
        M_per_sample: torch.Tensor,
        max_M: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract last M tokens per sample from last_hidden, right-aligned. Content at [L_max-L_per_sample, L_max)."""
        idx_grid = torch.arange(max_M, device=device)
        source_indices = L_max - M_per_sample.unsqueeze(1) + idx_grid.unsqueeze(0)
        valid_mask = idx_grid.unsqueeze(0) >= (max_M - M_per_sample.unsqueeze(1))
        safe_indices = source_indices.clamp(min=0, max=last_hidden.shape[1] - 1)
        gathered = torch.gather(
            last_hidden, 1,
            safe_indices.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1]),
        )
        doc_features = gathered * valid_mask.unsqueeze(-1).to(dtype=gathered.dtype)
        return doc_features, valid_mask

    def _extract_doc_features_from_last_hidden_memory_tokens(
        self,
        last_hidden: torch.Tensor,
        L_per_sample: torch.Tensor,
        M_per_sample: torch.Tensor,
        max_M: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract memory segment [L_per_sample, L_per_sample+M) from last_hidden, left-aligned."""
        idx_grid = torch.arange(max_M, device=device)
        source_indices = L_per_sample.unsqueeze(1) + idx_grid.unsqueeze(0)
        valid_mask = idx_grid.unsqueeze(0) < M_per_sample.unsqueeze(1)
        safe_indices = source_indices.clamp(max=last_hidden.shape[1] - 1)
        gathered = torch.gather(
            last_hidden, 1,
            safe_indices.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1]),
        )
        doc_features = gathered * valid_mask.unsqueeze(-1).to(dtype=gathered.dtype)
        return doc_features, valid_mask

    def _extract_doc_features_from_last_hidden_mean_pooling(
        self,
        last_hidden: torch.Tensor,
        L_max: int,
        L_per_sample: torch.Tensor,
        M_per_sample: torch.Tensor,
        pool_size_per_sample: torch.Tensor,
        max_M: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment mean pooling with variable pool_size per sample; result right-aligned."""
        seq_range = torch.arange(L_max, device=device).unsqueeze(0).expand(batch_size, -1)
        start_content = (L_max - L_per_sample).unsqueeze(1)
        rel_range = seq_range - start_content
        content_mask = (rel_range >= 0) & (rel_range < L_per_sample.unsqueeze(1))
        pool_indices = (rel_range // pool_size_per_sample.unsqueeze(1)).clamp(min=0)
        dest_indices = max_M - M_per_sample.unsqueeze(1) + pool_indices
        valid_dest = content_mask & (dest_indices >= 0) & (dest_indices < max_M)
        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * max_M
        flat_dest_indices = (batch_offsets + dest_indices).view(-1)
        flat_source = last_hidden.reshape(-1, last_hidden.shape[-1])
        flat_mask = valid_dest.view(-1)
        active_indices = flat_dest_indices[flat_mask]
        active_source = flat_source[flat_mask]
        output_flat = torch.zeros(batch_size * max_M, last_hidden.shape[-1], device=device, dtype=last_hidden.dtype)
        count_flat = torch.zeros(batch_size * max_M, 1, device=device, dtype=last_hidden.dtype)
        output_flat.index_add_(0, active_indices, active_source)
        count_flat.index_add_(0, active_indices, torch.ones(active_source.shape[0], 1, device=device, dtype=last_hidden.dtype))
        count_flat = count_flat.clamp(min=1.0)
        output_flat = output_flat / count_flat
        doc_features = output_flat.view(batch_size, max_M, -1)
        feature_idx = torch.arange(max_M, device=device).unsqueeze(0)
        doc_valid_mask = feature_idx >= (max_M - M_per_sample.unsqueeze(1))
        doc_features = doc_features * doc_valid_mask.unsqueeze(-1).to(dtype=doc_features.dtype)
        return doc_features, doc_valid_mask

    def _extract_doc_features_from_last_hidden_mean_pooling_fixed_pool_size(
        self,
        last_hidden: torch.Tensor,
        L_max: int,
        L_per_sample: torch.Tensor,
        pool_size: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed pool_size mean pooling: left-pad to multiple of pool_size, reshape and mean; result right-aligned."""
        hidden_dim = last_hidden.shape[-1]
        max_L = L_per_sample.max().item()
        content_list = []
        for b in range(batch_size):
            L_b = L_per_sample[b].item()
            c = last_hidden[b, L_max - L_b : L_max, :]
            if L_b < max_L:
                pad = torch.zeros(max_L - L_b, hidden_dim, dtype=last_hidden.dtype, device=device)
                c = torch.cat([pad, c], dim=0)
            content_list.append(c)
        content = torch.stack(content_list, dim=0)
        target_len = math.ceil(max_L / pool_size) * pool_size
        pad_len = target_len - max_L
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, hidden_dim, dtype=last_hidden.dtype, device=device)
            content = torch.cat([padding, content], dim=1)
        max_M = target_len // pool_size
        doc_features = content.view(batch_size, max_M, pool_size, hidden_dim).mean(dim=2)
        M_per_sample = ((L_per_sample + pool_size - 1) // pool_size).clamp(min=1)
        doc_valid_mask = torch.arange(max_M, device=device).unsqueeze(0) >= (max_M - M_per_sample.unsqueeze(1))
        doc_features = doc_features * doc_valid_mask.unsqueeze(-1).to(dtype=doc_features.dtype)
        return doc_features, doc_valid_mask

    def get_doc_features(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        return_last_hidden_state: bool = False,
    ):
        """
        Extract compressed document features from the encoder.

        Args:
            doc_input_ids: [batch_size, doc_len] token ids of the document.
            doc_attention_mask: Optional [batch_size, doc_len]; 1 for valid tokens. Used for per-sample length
                when using ratio-based or variable-length methods.
            return_last_hidden_state: If True, return a third value: encoder last_hidden_state [B, seq, embed_dim].

        Returns:
            doc_features: [batch_size, M_or_max, embed_dim] encoder-space compressed vectors.
            doc_valid_mask: [batch_size, M_or_max] True for valid positions (right-aligned when M varies).
            Optionally last_hidden_state when return_last_hidden_state=True.
        """
        batch_size = doc_input_ids.shape[0]
        device = doc_input_ids.device
        seq_len = doc_input_ids.shape[1]

        ratio_based = self.feature_extract_method in RATIO_BASED_FEATURE_EXTRACT_METHODS
        n_tokens = _comp_ratio_or_len_to_capacity(self.comp_ratio_or_len) if not ratio_based else None
        if n_tokens is not None and seq_len < n_tokens:
            raise ValueError(f"seq_len ({seq_len}) must be >= comp_ratio_or_len capacity ({n_tokens})")

        def _return_two_or_three(doc_features, doc_valid_mask, last_hidden):
            if return_last_hidden_state:
                return doc_features, doc_valid_mask, last_hidden
            return doc_features, doc_valid_mask

        model_input_ids, model_attention_mask, L_per_sample = self._prepare_doc_encoder_inputs(
            doc_input_ids, doc_attention_mask, batch_size, device,
        )
        last_hidden = self._get_doc_encoder_last_hidden(model_input_ids, model_attention_mask)
        L_max = last_hidden.shape[1]

        if self.feature_extract_method == "last_tokens":
            n = int(self.comp_ratio_or_len)
            M_per_sample = torch.full((batch_size,), n, dtype=torch.long, device=device)
            max_M = n
            df, dm = self._extract_doc_features_from_last_hidden_last_tokens(
                last_hidden, L_max, L_per_sample, M_per_sample, max_M, batch_size, device,
            )
            return _return_two_or_three(df, dm, last_hidden)
        if self.feature_extract_method in ("same_memory_tokens", "different_memory_tokens"):
            n = int(self.comp_ratio_or_len)
            M_per_sample = torch.full((batch_size,), n, dtype=torch.long, device=device)
            max_M = n
            df, dm = self._extract_doc_features_from_last_hidden_memory_tokens(
                last_hidden, L_per_sample, M_per_sample, max_M, batch_size, device,
            )
            return _return_two_or_three(df, dm, last_hidden)
        if self.feature_extract_method in ("mean_pooling", "mean_pooling_causal"):
            ratio = float(self.comp_ratio_or_len)
            pool_size = max(1, round(1 / ratio))
            df, dm = self._extract_doc_features_from_last_hidden_mean_pooling_fixed_pool_size(
                last_hidden, L_max, L_per_sample, pool_size, batch_size, device,
            )
            return _return_two_or_three(df, dm, last_hidden)
        raise ValueError(
            f"Unsupported feature_extract_method: {self.feature_extract_method}. "
            f"Supported: {SUPPORTED_FEATURE_EXTRACT_METHODS}"
        )

    def compute_inputs_embeds(
        self,
        doc_input_ids: torch.Tensor,
        input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute decoder input embeddings with document context injected at placeholders.

        Args:
            doc_input_ids: Document token ids [batch_size, doc_len].
            input_ids: Decoder query/prompt token ids [batch_size, seq_len], containing placeholder_token_id.
            doc_attention_mask: Optional attention mask for the document.
            attention_mask: Optional attention mask for the query (decoder).

        Returns:
            inputs_embeds: [batch_size, seq_len, hidden_dim] ready for decoder(inputs_embeds=..., attention_mask=...).
        """
        doc_features, doc_valid_mask = self.get_doc_features(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
        )
        compressed_doc_features = self.projector(doc_features)
        return self.replace_placeholder_tokens(
            input_ids=input_ids,
            compressed_doc_features=compressed_doc_features,
            doc_valid_mask=doc_valid_mask,
            doc_attention_mask=doc_attention_mask,
            doc_input_ids=doc_input_ids,
        )

    def forward(
        self,
        doc_input_ids: torch.Tensor,
        input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        comp_ratio_or_len_override: Optional[Union[int, float]] = None,
    ):
        """
        Forward pass: compress document, inject at placeholders, run decoder with optional labels.

        Args:
            doc_input_ids: Document token ids [batch_size, doc_len].
            input_ids: Decoder input token ids [batch_size, seq_len] with placeholder_token_id.
            doc_attention_mask: Optional document attention mask.
            attention_mask: Optional decoder attention mask.
            labels: Optional [batch_size, seq_len] for language modeling loss (-100 to ignore positions).
            comp_ratio_or_len_override: When the model has multiple comp_ratio_or_len options (tuple/list),
                pass the value to use for this batch. Ignored when the model has a single fixed value.

        Returns:
            Decoder output (e.g. CausalLMOutputWithPast with loss, logits, etc.).
        """
        to_set = _validate_forward_comp_ratio_or_len(self.comp_ratio_or_len, comp_ratio_or_len_override)
        _saved = None
        if to_set is not None:
            _saved = self.comp_ratio_or_len
            self.comp_ratio_or_len = to_set
        try:
            inputs_embeds = self.compute_inputs_embeds(
                doc_input_ids=doc_input_ids,
                input_ids=input_ids,
                doc_attention_mask=doc_attention_mask,
                attention_mask=attention_mask,
            )
            self._switch_decoder_adapter()
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            return outputs
        finally:
            if _saved is not None:
                self.comp_ratio_or_len = _saved

    def save_projector(self, path: str) -> None:
        """Save only the projector MLP state dict to path (e.g. projector.pth)."""
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str) -> None:
        """Load projector MLP state dict from path. Device is taken from current projector parameters.
        Supports multiple state_dict formats:
        - Direct: keys like up_proj.weight, down_proj.weight, rms_norm.weight (from save_projector).
        - Nested: {"projector": {...}, "compress_ratio_head": ...} (e.g. from CtxCompSemiDynamicModel).
        - Prefixed: keys like "projector.up_proj.weight" (e.g. from full model state_dict).
        """
        device = self.projector.down_proj.weight.device
        try:
            raw = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            raw = torch.load(path, map_location=device)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected state_dict dict at {path}, got {type(raw).__name__}")
        # Nested: {"projector": {...}, "compress_ratio_head": ...}
        if "projector" in raw and isinstance(raw["projector"], dict):
            state = raw["projector"]
        # Flat with prefix: {"projector.up_proj.weight": ..., ...}
        elif any(k.startswith("projector.") for k in raw):
            prefix = "projector."
            state = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
        else:
            state = raw
        # Load only keys that exist in this projector (ignore compress_ratio_head etc.)
        model_keys = set(self.projector.state_dict().keys())
        to_load = {k: v for k, v in state.items() if k in model_keys}
        if not to_load:
            raise ValueError(
                f"No projector keys found in {path}. Model expects {list(model_keys)}, "
                f"state has keys {list(state.keys())}"
            )
        self.projector.load_state_dict(to_load, strict=False)

    def generate(
        self,
        doc_input_ids: torch.Tensor,
        input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        comp_ratio_or_len_override: Optional[Union[int, float]] = None,
        *args,
        **kwargs,
    ):
        """
        Generate with compressed document context. Passes *args and **kwargs to decoder.generate.

        Args:
            doc_input_ids: Document token ids [batch_size, doc_len].
            input_ids: Decoder prompt token ids [batch_size, seq_len] with placeholder_token_id.
            doc_attention_mask: Optional document attention mask.
            attention_mask: Optional decoder attention mask for the prompt.
            comp_ratio_or_len_override: When model has multiple comp_ratio_or_len options, the value for this call.

        Returns:
            Generated token ids or sequences (same as decoder.generate).
        """
        to_set = _validate_forward_comp_ratio_or_len(self.comp_ratio_or_len, comp_ratio_or_len_override)
        _saved = None
        if to_set is not None:
            _saved = self.comp_ratio_or_len
            self.comp_ratio_or_len = to_set
        try:
            with torch.no_grad():
                inputs_embeds = self.compute_inputs_embeds(
                    doc_input_ids=doc_input_ids,
                    doc_attention_mask=doc_attention_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                self._switch_decoder_adapter()
                return self.decoder.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    *args,
                    **kwargs,
                )
        finally:
            if _saved is not None:
                self.comp_ratio_or_len = _saved

    def _set_model_trainable(
        self,
        model,
        training_mode: str,
        lora_config: Optional[LoraConfig] = None,
        model_name: str = "model",
        is_causal_lm: bool = False,
    ):
        """Set model training mode: full, lora, or none."""
        if training_mode == "full":
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        elif training_mode == "lora":
            if lora_config is None:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=target_modules,
                    lora_dropout=0.1,
                    bias="none",
                    use_rslora=True,
                    task_type="CAUSAL_LM" if is_causal_lm else None,
                )
            model = get_peft_model(model, lora_config)
        elif training_mode == "none":
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Invalid {model_name}_training: {training_mode}. Must be 'full', 'lora', or 'none'")
        return model

    def set_trainable_params(
        self,
        encoder_lora_config: Optional[LoraConfig] = None,
        decoder_lora_config: Optional[LoraConfig] = None,
    ) -> None:
        """
        Set which parameters are trainable based on encoder_training and decoder_training.

        The projector is always set to trainable. For encoder/decoder, "full" = all params,
        "lora" = wrap with LoRA (using provided config or default), "none" = freeze.
        When using memory_tokens methods and encoder is trained, only memory token embedding rows are trainable.

        Args:
            encoder_lora_config: LoRA config for encoder when encoder_training=="lora". If None, a default is used.
            decoder_lora_config: LoRA config for decoder when decoder_training=="lora". If None, a default is used.
        """
        self.projector.train()
        for param in self.projector.parameters():
            param.requires_grad = True

        self.encoder = self._set_model_trainable(
            self.encoder,
            self.encoder_training,
            encoder_lora_config,
            model_name="encoder",
            is_causal_lm=False,
        )
        if (
            self.encoder_training != "none"
            and self.feature_extract_method in ("same_memory_tokens", "different_memory_tokens")
        ):
            self._set_memory_token_embeddings_trainable(self.encoder)

        self.decoder = self._set_model_trainable(
            self.decoder,
            self.decoder_training,
            decoder_lora_config,
            model_name="decoder",
            is_causal_lm=True,
        )

    def gradient_checkpointing_enable(self,*args,**kwargs) -> None:
        """Enable gradient checkpointing on encoder and decoder to save memory. Use with ddp_find_unused_parameters=True."""
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(*args,**kwargs)
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable(*args,**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on encoder and decoder."""
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
        if hasattr(self.decoder, "gradient_checkpointing_disable"):
            self.decoder.gradient_checkpointing_disable()

    def get_nb_trainable_parameters(self) -> Tuple[int, int]:
        """Return (trainable_params, all_params)."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")

    def _get_save_config(self) -> dict:
        """Build config dict for config.json."""
        def get_base_model_path(m):
            if isinstance(m, PeftModel):
                return m.get_base_model().config.name_or_path
            return m.config.name_or_path

        return {
            "comp_ratio_or_len": _comp_ratio_or_len_to_serializable(self.comp_ratio_or_len),
            "placeholder_token_id": self.placeholder_token_id,
            "mlp_converter_hidden_dim": self.projector.intermediate_size,
            "feature_extract_method": self.feature_extract_method,
            "memory_token_begin_id": self.memory_token_begin_id,
            "encoder_training": self.encoder_training,
            "decoder_training": self.decoder_training,
            "base_encoder_model_path": get_base_model_path(self.encoder),
            "base_decoder_model_path": get_base_model_path(self.decoder),
        }

    def save_pretrained(self, path: str) -> None:
        """
        Save model to a directory for later loading with from_pretrained.

        Saves: config.json; encoder/ (if encoder_training != "none"); decoder/ (if decoder_training != "none");
        projector.pth; and memory_token_embedding.pth when using memory_tokens and encoder is trained.
        You can customize what gets saved by setting encoder_training/decoder_training before calling.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        if self.encoder_training != "none":
            self.encoder.save_pretrained(os.path.join(path, "encoder"))
        if self.decoder_training != "none":
            self.decoder.save_pretrained(os.path.join(path, "decoder"))
        self.save_projector(os.path.join(path, "projector.pth"))

        if self._memory_token_indices and self.encoder_training != "none":
            embed_tokens = _get_embed_tokens_module(self.encoder)
            if embed_tokens is not None:
                indices = self._memory_token_indices
                memory_embed_state = {
                    "indices": indices,
                    "weight": embed_tokens.weight.data[indices].detach().cpu().clone(),
                }
                torch.save(memory_embed_state, os.path.join(path, "memory_token_embedding.pth"))

        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self._get_save_config(), f, indent=4)

    @classmethod
    def from_pretrained(cls, path: str, share_base_model_inference: bool = False, **kwargs):
        """
        Load model from a directory saved by save_pretrained.

        Args:
            path: Directory containing config.json, encoder/, decoder/, projector.pth (and optionally
                memory_token_embedding.pth). Backward compatible with "embed_model"/"gen_model" subdir names.
            share_base_model_inference: If True, and both encoder and decoder were trained with LoRA and
                share the same base model path, loads the base model once and attaches both LoRA adapters.
                Saves GPU memory at inference by switching adapters when using encoder vs decoder. Not used
                for training (train with separate encoder/decoder as usual).
            **kwargs: Passed to AutoModel / AutoModelForCausalLM / PeftModel (e.g. torch_dtype, device_map).

        Returns:
            CtxCompModel instance with weights and config restored.
        """
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        encoder_training = config.get("encoder_training", config.get("emb_model_training", "none"))
        decoder_training = config.get("decoder_training", config.get("gen_model_training", "none"))
        base_encoder_path = config.get("base_encoder_model_path", config.get("base_embed_model_path"))
        base_decoder_path = config.get("base_decoder_model_path", config.get("base_gen_model_path"))
        encoder_path = os.path.join(path, "encoder")
        decoder_path = os.path.join(path, "decoder")
        if not os.path.isdir(encoder_path):
            encoder_path = os.path.join(path, "embed_model")
        if not os.path.isdir(decoder_path):
            decoder_path = os.path.join(path, "gen_model")

        encoder = None
        decoder = None
        shared_base = False

        if share_base_model_inference and encoder_training == "lora" and decoder_training == "lora" and base_encoder_path == base_decoder_path:
            base_model = AutoModelForCausalLM.from_pretrained(base_decoder_path, **kwargs)
            base_model = PeftModel.from_pretrained(base_model, decoder_path, adapter_name="decoder")

            base_model.load_adapter(encoder_path, adapter_name="encoder")
            encoder = decoder = base_model
            shared_base = True

        else:
            if encoder_training == "full":
                encoder = AutoModel.from_pretrained(encoder_path, **kwargs)
            elif encoder_training == "lora":
                encoder = AutoModel.from_pretrained(base_encoder_path, **kwargs)
                encoder = PeftModel.from_pretrained(encoder, encoder_path)
            else:
                encoder = AutoModel.from_pretrained(base_encoder_path, **kwargs)

            if decoder_training == "full":
                decoder = AutoModelForCausalLM.from_pretrained(decoder_path, **kwargs)
            elif decoder_training == "lora":
                decoder = AutoModelForCausalLM.from_pretrained(base_decoder_path, **kwargs)
                decoder = PeftModel.from_pretrained(decoder, decoder_path)
            else:
                decoder = AutoModelForCausalLM.from_pretrained(base_decoder_path, **kwargs)

        comp_ratio_or_len = config.get("comp_ratio_or_len", config.get("num_doc_tokens", 1))
        feature_extract_method = config.get("feature_extract_method", config.get("compress_method", "last_tokens"))
        if feature_extract_method in ("mean_pooling_by_ratio", "mean_pooling"):
            feature_extract_method = "mean_pooling"
        if feature_extract_method in ("mean_pooling_by_ratio_causal", "mean_pooling_causal"):
            feature_extract_method = "mean_pooling_causal"

        model = cls(
            encoder=encoder,
            decoder=decoder,
            placeholder_token_id=config["placeholder_token_id"],
            comp_ratio_or_len=comp_ratio_or_len,
            mlp_converter_hidden_dim=config.get("mlp_converter_hidden_dim", config.get("mlp_hidden_dim", 4096)),
            feature_extract_method=feature_extract_method,
            memory_token_begin_id=config.get("memory_token_begin_id", 151000),
            encoder_training=encoder_training,
            decoder_training=decoder_training,
        )
        model._shared_base_model = shared_base
        model.load_projector(os.path.join(path, "projector.pth"))

        memory_embed_path = os.path.join(path, "memory_token_embedding.pth")
        if os.path.isfile(memory_embed_path):
            try:
                memory_embed_state = torch.load(memory_embed_path, map_location="cpu", weights_only=True)
            except TypeError:
                memory_embed_state = torch.load(memory_embed_path, map_location="cpu")
            indices = memory_embed_state["indices"]
            weight_slice = memory_embed_state["weight"]
            embed_tokens = _get_embed_tokens_module(model.encoder)
            if embed_tokens is not None:
                target_weight = embed_tokens.weight
                with torch.no_grad():
                    target_weight[indices] = weight_slice.to(device=target_weight.device, dtype=target_weight.dtype)

        return model

    def reinit_memory_tokens(self) -> None:
        """
        Reinitialize the embedding rows for memory tokens (same_memory_tokens / different_memory_tokens).
        Useful when you want to retrain only the memory tokens from scratch. No-op for other feature_extract_methods.
        """
        indices = self._get_memory_token_indices()
        if not indices:
            return
        embed_tokens = _get_embed_tokens_module(self.encoder)
        if embed_tokens is not None:
            with torch.no_grad():
                embed_tokens.weight[indices] = torch.randn_like(embed_tokens.weight[indices])

    def load_from_checkpoint(self, path: str, **kwargs) -> None:
        """
        Load weights from a checkpoint directory into this existing instance (for training resume).

        Expects the same layout as save_pretrained (config.json, encoder/, decoder/, projector.pth, etc.).
        LoRA adapters are loaded with load_adapter so optimizer parameter groups stay consistent.
        """
        path = os.fspath(path)
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        encoder_training = config.get("encoder_training", config.get("emb_model_training", "none"))
        decoder_training = config.get("decoder_training", config.get("gen_model_training", "none"))
        encoder_path = os.path.join(path, "encoder")
        decoder_path = os.path.join(path, "decoder")
        if not os.path.isdir(encoder_path):
            encoder_path = os.path.join(path, "embed_model")
        if not os.path.isdir(decoder_path):
            decoder_path = os.path.join(path, "gen_model")

        def _load_state_dict_from_dir(dir_path: str) -> dict:
            try:
                import safetensors.torch as st
            except ImportError:
                st = None
            safe_path = os.path.join(dir_path, "adapter_model.safetensors")
            bin_path = os.path.join(dir_path, "adapter_model.bin")
            full_safe = os.path.join(dir_path, "model.safetensors")
            full_bin = os.path.join(dir_path, "pytorch_model.bin")
            if st is not None and os.path.isfile(safe_path):
                return dict(st.load_file(safe_path))
            if os.path.isfile(bin_path):
                state = torch.load(bin_path, map_location="cpu", weights_only=True)
                return state if isinstance(state, dict) else {}
            if st is not None and os.path.isfile(full_safe):
                return dict(st.load_file(full_safe))
            if os.path.isfile(full_bin):
                state = torch.load(full_bin, map_location="cpu", weights_only=True)
                return state if isinstance(state, dict) else {}
            return {}

        if encoder_training != "none" and os.path.isdir(encoder_path):
            if encoder_training == "lora" and isinstance(self.encoder, PeftModel):
                for name in list(self.encoder.peft_config.keys()):
                    self.encoder.delete_adapter(name)
                self.encoder.load_adapter(encoder_path, "default", is_trainable=True)
                self.encoder.set_adapter("default")
            else:
                state = _load_state_dict_from_dir(encoder_path)
                if state:
                    self.encoder.load_state_dict(state, strict=False)

        if decoder_training != "none" and os.path.isdir(decoder_path):
            if decoder_training == "lora" and isinstance(self.decoder, PeftModel):
                for name in list(self.decoder.peft_config.keys()):
                    self.decoder.delete_adapter(name)
                self.decoder.load_adapter(decoder_path, "default", is_trainable=True)
                self.decoder.set_adapter("default")
            else:
                state = _load_state_dict_from_dir(decoder_path)
                if state:
                    self.decoder.load_state_dict(state, strict=False)

        projector_path = os.path.join(path, "projector.pth")
        if os.path.isfile(projector_path):
            self.load_projector(projector_path)

        memory_embed_path = os.path.join(path, "memory_token_embedding.pth")
        if os.path.isfile(memory_embed_path):
            try:
                memory_embed_state = torch.load(memory_embed_path, map_location="cpu", weights_only=True)
            except TypeError:
                memory_embed_state = torch.load(memory_embed_path, map_location="cpu")
            indices = memory_embed_state["indices"]
            weight_slice = memory_embed_state["weight"]
            embed_tokens = _get_embed_tokens_module(self.encoder)
            if embed_tokens is not None:
                with torch.no_grad():
                    embed_tokens.weight[indices] = weight_slice.to(device=embed_tokens.weight.device, dtype=embed_tokens.weight.dtype)


class CtxCompSemiDynamicModel(CtxCompModel):
    """
    Semi-dynamic context compression: compression length M can be chosen from discrete options at inference.

    comp_ratio_or_len must be a non-empty list/tuple of allowed values (ints or ratios). A small head
    (compress_ratio_head) predicts log2(context_length/summary_length); the prediction is discretized to
    the nearest option to get M. Training uses compress_len_labels in the same log space (MSE loss).

    Two modes:
    - With comp_ratio_or_len_override set: fixed M for the call; same as CtxCompModel (input must have M placeholders).
    - With comp_ratio_or_len_override=None (e.g. inference): exactly one placeholder per sample; it is expanded
      to M positions where M is predicted and discretized per sample.
    """

    def __init__(
        self,
        encoder,
        decoder,
        placeholder_token_id: int,
        memory_token_begin_id: int = 151000,
        comp_ratio_or_len: Union[Tuple[Union[int, float], ...], List[Union[int, float]]] = None,
        mlp_converter_hidden_dim: int = 4096,
        encoder_training: str = "none",
        decoder_training: str = "none",
        feature_extract_method: str = "last_tokens",
        discretize_ratio_mode: str = "round",
        discretize_compare_in_log: bool = False,
    ):
        """
        Args:
            encoder, decoder, placeholder_token_id, memory_token_begin_id, mlp_converter_hidden_dim,
            encoder_training, decoder_training: Same as CtxCompModel (see CtxCompModel.__init__).
            comp_ratio_or_len: Non-empty list or tuple of allowed compression values. All ints (fixed M options)
                or all floats in (0, 1] (ratio options). At inference, the head prediction is discretized to
                the nearest value in this list.
            feature_extract_method: One of the supported methods; must match the type of comp_ratio_or_len
                (e.g. use "mean_pooling" or "mean_pooling_causal" for float ratios).
            discretize_ratio_mode: When discretizing the predicted ratio to comp_ratio_or_len: "round" (nearest),
                "ceil" (choose smallest option >= prediction), or "floor" (largest option <= prediction).
            discretize_compare_in_log: If True, discretization is done in log2(1/ratio) space; if False, in 1/ratio
                space. Usually True for consistency with compress_len_labels = log2(context/summary).
        """
        if not isinstance(comp_ratio_or_len, (list, tuple)) or len(comp_ratio_or_len) == 0:
            raise ValueError("CtxCompSemiDynamicModel requires comp_ratio_or_len to be a non-empty list or tuple.")
        comp_ratio_or_len = tuple(comp_ratio_or_len)
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            placeholder_token_id=placeholder_token_id,
            memory_token_begin_id=memory_token_begin_id,
            comp_ratio_or_len=comp_ratio_or_len,
            mlp_converter_hidden_dim=mlp_converter_hidden_dim,
            feature_extract_method=feature_extract_method,
            encoder_training=encoder_training,
            decoder_training=decoder_training,
        )
        self.discretize_ratio_mode = discretize_ratio_mode
        self.discretize_compare_in_log = discretize_compare_in_log
        embed_dim = _model_hidden_size(self.encoder)
        self.compress_ratio_head = nn.Linear(
            embed_dim, 1, device=decoder.device, dtype=decoder.dtype,
        )

    def _discretize(self, values: torch.Tensor, available_values: list) -> torch.Tensor:
        """
        Discretize each element of values to the nearest value in available_values (by linear distance).
        Used when comp_ratio_or_len is a list of integers (target M); raw M is rounded to nearest allowed.
        """
        if not available_values:
            return values
        device, dtype = values.device, values.dtype
        candidates = torch.tensor(available_values, device=device, dtype=dtype)
        v_expanded = values.unsqueeze(-1) if values.dim() == 1 else values.unsqueeze(-1)
        diff = (v_expanded - candidates.unsqueeze(0)).abs()
        min_indices = diff.argmin(dim=-1)
        nearest = candidates[min_indices]
        if values.dim() == 2:
            nearest = nearest.unsqueeze(-1)
        return nearest

    def _discretize_ratio_log(
        self,
        values: torch.Tensor,
        available_ratios: List[float],
        mode: str = "round",
        compare_in_log: bool = True,
        values_in_log_space: bool = False,
    ) -> torch.Tensor:
        """
        Discretize predicted ratio (or log) to the nearest in available_ratios.
        values: [B] or [B,1]; if values_in_log_space=True they are log2(context/summary), else ratio in (0,1].
        compare_in_log: if True, compare in log2(1/r) space; else in 1/r space.
        Returns tensor of same shape as values with discretized ratios.
        """
        device = values.device
        dtype = values.dtype
        flat = values.view(-1)
        if compare_in_log:
            candidate_log = torch.tensor(
                [math.log2(1.0 / r) for r in available_ratios],
                device=device,
                dtype=dtype,
            )
            target = flat if values_in_log_space else torch.log2(1.0 / flat.clamp(min=1e-9))
        else:
            candidate_log = torch.tensor(
                [1.0 / r for r in available_ratios],
                device=device,
                dtype=dtype,
            )
            target = torch.pow(2.0, flat) if values_in_log_space else flat
        diff_signed = target.unsqueeze(1) - candidate_log.unsqueeze(0)
        inf_t = torch.full_like(diff_signed, float("inf"))
        if mode == "round":
            diff_compare = diff_signed.abs()
        elif mode == "ceil":
            diff_compare = torch.where(diff_signed < 0, diff_signed.abs(), inf_t)
        elif mode == "floor":
            diff_compare = torch.where(diff_signed > 0, diff_signed.abs(), inf_t)
        else:
            raise ValueError(f"discretize_ratio_mode must be 'round', 'ceil' or 'floor', got {mode!r}")
        min_idx = diff_compare.argmin(dim=1)
        if mode == "ceil":
            no_candidate = (diff_compare == float("inf")).all(dim=1)
            min_idx = torch.where(no_candidate, diff_signed.argmax(dim=1), min_idx)
        elif mode == "floor":
            no_candidate = (diff_compare == float("inf")).all(dim=1)
            min_idx = torch.where(no_candidate, diff_signed.argmin(dim=1), min_idx)
        candidates = torch.tensor(available_ratios, device=device, dtype=dtype)
        out_flat = candidates[min_idx]
        return out_flat.view_as(values)

    def _get_doc_valid_lengths(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-sample valid document length L from doc_attention_mask."""
        L_max = doc_input_ids.shape[1]
        if doc_attention_mask is not None:
            return doc_attention_mask.sum(dim=1).clamp(min=1)
        return torch.full((batch_size,), L_max, dtype=torch.long, device=device)

    def _get_max_memory_slots(self) -> int:
        """Max memory token slots: max of comp_ratio_or_len if all int, else vocab range."""
        if all(isinstance(r, int) for r in self.comp_ratio_or_len):
            return max(self.comp_ratio_or_len)
        embed_tokens = _get_embed_tokens_module(self.encoder)
        if embed_tokens is None:
            return max(self.comp_ratio_or_len) if self.comp_ratio_or_len else 0
        vocab_size = embed_tokens.weight.shape[0]
        return max(0, vocab_size - self.memory_token_begin_id)

    def get_doc_features(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        comp_ratio_or_len_override: Optional[Union[int, float]] = None,
        compress_ratio_scale: float = 0,
    ):
        """
        Extract document features; M is either fixed (override) or predicted by compress_ratio_head and discretized.

        Args:
            doc_input_ids: Document token ids [batch_size, doc_len].
            doc_attention_mask: Optional; required for per-sample length when M is predicted.
            comp_ratio_or_len_override: If set, use this as fixed M (same as base class). If None, M is
                predicted from the last token hidden state and discretized to comp_ratio_or_len options.
            compress_ratio_scale: Added to the head output (log space) before discretization. Use to shift
                predicted compression ratio at inference without retraining (e.g. 0.5 for more compression).

        Returns:
            doc_features: [batch_size, max_M, embed_dim].
            doc_valid_mask: [batch_size, max_M] True for valid positions.
            predicted_compress_ratio: [batch_size, 1] head output (log space); used for comp_ratio_loss when labels given.
        """
        if comp_ratio_or_len_override is not None:
            batch_size = doc_input_ids.shape[0]
            device = doc_input_ids.device
            _saved = self.comp_ratio_or_len
            self.comp_ratio_or_len = comp_ratio_or_len_override
            try:
                doc_features, doc_valid_mask, last_hidden_state = super().get_doc_features(
                    doc_input_ids=doc_input_ids,
                    doc_attention_mask=doc_attention_mask,
                    return_last_hidden_state=True,
                )
            finally:
                self.comp_ratio_or_len = _saved
            L_per_sample = self._get_doc_valid_lengths(doc_input_ids, doc_attention_mask, batch_size, device)
            last_content_idx = _last_content_indices_from_attention_mask(doc_attention_mask, L_per_sample, device)
            last_token_hidden = last_hidden_state[
                torch.arange(batch_size, device=device), last_content_idx, :
            ]
            predicted_compress_ratio = self.compress_ratio_head(last_token_hidden) + compress_ratio_scale
            return doc_features, doc_valid_mask, predicted_compress_ratio

        batch_size = doc_input_ids.shape[0]
        device = doc_input_ids.device
        L_max = doc_input_ids.shape[1]
        is_memory = self.feature_extract_method in ("same_memory_tokens", "different_memory_tokens")
        is_mean_pooling = self.feature_extract_method in ("mean_pooling", "mean_pooling_causal")

        num_for_prepare = _comp_ratio_or_len_for_max_memory(self.comp_ratio_or_len)
        model_input_ids, model_attention_mask, L_per_sample = self._prepare_doc_encoder_inputs(
            doc_input_ids, doc_attention_mask, batch_size, device, comp_ratio_or_len_arg=num_for_prepare,
        )
        last_hidden = self._get_doc_encoder_last_hidden(model_input_ids, model_attention_mask)
        last_content_idx = _last_content_indices_from_attention_mask(doc_attention_mask, L_per_sample, device)
        last_token_hidden = last_hidden[torch.arange(batch_size, device=device), last_content_idx, :]
        predicted_compress_ratio = self.compress_ratio_head(last_token_hidden) + compress_ratio_scale
        pred_log = predicted_compress_ratio.squeeze(1)

        ratio_based = isinstance(self.comp_ratio_or_len[0], float)
        pool_size_per_sample = None
        if ratio_based:
            ratio_discrete = self._discretize_ratio_log(
                pred_log, list(self.comp_ratio_or_len),
                mode=self.discretize_ratio_mode,
                compare_in_log=self.discretize_compare_in_log,
                values_in_log_space=True,
            )
            if is_mean_pooling:
                pool_size_per_sample = torch.round(1.0 / ratio_discrete).long().clamp(min=1)
                M_per_sample = torch.ceil(L_per_sample.float() / pool_size_per_sample.float()).long()
            else:
                M_per_sample = (L_per_sample.float() * ratio_discrete).long().clamp(min=1)
        else:
            ratio_continuous = torch.pow(2.0, -pred_log).clamp(min=1e-9, max=1.0)
            raw_M = (L_per_sample.float() * ratio_continuous).long()
            M_per_sample = self._discretize(raw_M, list(self.comp_ratio_or_len)).long()

        M_per_sample = torch.minimum(M_per_sample, L_per_sample).clamp(min=1)
        max_M = max(M_per_sample.max().item(), 1)

        if is_memory:
            doc_features, doc_valid_mask = self._extract_doc_features_from_last_hidden_memory_tokens(
                last_hidden, L_per_sample, M_per_sample, max_M, batch_size, device,
            )
        elif is_mean_pooling:
            if pool_size_per_sample is None:
                pool_size_per_sample = torch.ceil(L_per_sample.float() / M_per_sample.float()).long().clamp(min=1)
            doc_features, doc_valid_mask = self._extract_doc_features_from_last_hidden_mean_pooling(
                last_hidden, L_max, L_per_sample, M_per_sample, pool_size_per_sample,
                max_M, batch_size, device,
            )
        else:
            doc_features, doc_valid_mask = self._extract_doc_features_from_last_hidden_last_tokens(
                last_hidden, L_max, L_per_sample, M_per_sample, max_M, batch_size, device,
            )
        return doc_features, doc_valid_mask, predicted_compress_ratio

    def replace_placeholder_tokens_expand_one(
        self,
        input_ids: torch.Tensor,
        compressed_doc_features: torch.Tensor,
        doc_valid_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], bool]:
        """Replace the single placeholder per sample with M positions. Returns (inputs_embeds, new_attention_mask, placeholder_pos, new_len_list, pad_on_right)."""
        if attention_mask is not None:
            pad_on_right = not (attention_mask[:, 0] == 0).any().item()
        else:
            pad_on_right = True
        base_embeds = self.decoder.get_input_embeddings()(input_ids)
        batch_size, S, hidden_dim = base_embeds.shape
        device = input_ids.device
        placeholder_mask = (input_ids == self.placeholder_token_id)
        n_placeholders = placeholder_mask.sum(dim=1)
        for i in range(batch_size):
            if n_placeholders[i].item() != 1:
                raise ValueError(
                    f"Sample {i}: expected exactly 1 placeholder (id={self.placeholder_token_id}), "
                    f"found {n_placeholders[i].item()}."
                )
        placeholder_pos = placeholder_mask.float().argmax(dim=1)
        new_seq_list, new_mask_list = [], []
        for b in range(batch_size):
            p = int(placeholder_pos[b].item())
            injected = compressed_doc_features[b][doc_valid_mask[b]]
            M_b = injected.shape[0]
            prefix = base_embeds[b, :p]
            suffix = base_embeds[b, p + 1:]
            new_seq_b = torch.cat([prefix, injected, suffix], dim=0)
            new_seq_list.append(new_seq_b)
            if attention_mask is not None:
                new_mask_b = torch.cat([
                    attention_mask[b, :p],
                    torch.ones(M_b, dtype=attention_mask.dtype, device=device),
                    attention_mask[b, p + 1:],
                ], dim=0)
            else:
                new_mask_b = torch.ones(new_seq_b.shape[0], dtype=torch.long, device=device)
            new_mask_list.append(new_mask_b)
        new_len_list = [s.shape[0] for s in new_seq_list]
        max_len = max(new_len_list)
        padded_embeds, new_attention_mask_list = [], []
        for b in range(batch_size):
            seq = new_seq_list[b]
            mask_b = new_mask_list[b]
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                padding = torch.zeros((pad_len, hidden_dim), dtype=seq.dtype, device=device)
                pad_mask = torch.zeros(pad_len, dtype=mask_b.dtype, device=device)
                if pad_on_right:
                    seq = torch.cat([seq, padding], dim=0)
                    mask_b = torch.cat([mask_b, pad_mask], dim=0)
                else:
                    seq = torch.cat([padding, seq], dim=0)
                    mask_b = torch.cat([pad_mask, mask_b], dim=0)
            padded_embeds.append(seq)
            new_attention_mask_list.append(mask_b)
        inputs_embeds = torch.stack(padded_embeds, dim=0)
        new_attention_mask = torch.stack(new_attention_mask_list, dim=0)
        return inputs_embeds, new_attention_mask, placeholder_pos, new_len_list, pad_on_right

    def _expand_labels(
        self,
        labels: torch.Tensor,
        placeholder_pos: torch.Tensor,
        num_valid_tokens: torch.Tensor,
        new_len_list: list,
        max_len: int,
        pad_on_right: bool,
    ) -> torch.Tensor:
        """Expand labels [B, S] to [B, max_len] with -100 at inserted M positions and padding."""
        batch_size, S = labels.shape
        device = labels.device
        expanded_list = []
        for b in range(batch_size):
            p = int(placeholder_pos[b].item())
            M_b = int(num_valid_tokens[b].item())
            left = labels[b, :p]
            inserted = torch.full((M_b,), -100, dtype=labels.dtype, device=device)
            right = labels[b, p + 1:]
            new_labels_b = torch.cat([left, inserted, right], dim=0)
            pad_len = max_len - new_labels_b.shape[0]
            if pad_len > 0:
                pad_labels = torch.full((pad_len,), -100, dtype=labels.dtype, device=device)
                new_labels_b = torch.cat([new_labels_b, pad_labels], dim=0) if pad_on_right else torch.cat([pad_labels, new_labels_b], dim=0)
            expanded_list.append(new_labels_b)
        return torch.stack(expanded_list, dim=0)

    def set_trainable_params(
        self,
        encoder_lora_config: Optional[LoraConfig] = None,
        decoder_lora_config: Optional[LoraConfig] = None,
    ) -> None:
        """Same as parent and set compress_ratio_head trainable."""
        self.compress_ratio_head.train()
        for param in self.compress_ratio_head.parameters():
            param.requires_grad = True
        super().set_trainable_params(encoder_lora_config=encoder_lora_config, decoder_lora_config=decoder_lora_config)

    def compute_inputs_embeds(
        self,
        doc_input_ids: torch.Tensor,
        input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        comp_ratio_or_len_override: Optional[Union[int, float]] = None,
        compress_ratio_scale: float = 0,
    ):
        """
        Compute decoder input embeddings. When comp_ratio_or_len_override is None, each sample must have
        exactly one placeholder; it is expanded to M positions (M predicted per sample).

        Returns a 7-tuple: (inputs_embeds, attention_mask, placeholder_pos, num_valid_tokens, new_len_list,
        predicted_compress_ratio, pad_on_right_or_None). When override is set, placeholder_pos and related are None.
        """
        doc_features, doc_valid_mask, predicted_compress_ratio = self.get_doc_features(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            comp_ratio_or_len_override=comp_ratio_or_len_override,
            compress_ratio_scale=compress_ratio_scale,
        )
        compressed_doc_features = self.projector(doc_features)

        # if comp_ratio_or_len_override is not None, that means the fixed-ratio mode, so we use the parent class's replace_placeholder_tokens
        if comp_ratio_or_len_override is not None:
            inputs_embeds = super().replace_placeholder_tokens(
                input_ids=input_ids,
                compressed_doc_features=compressed_doc_features,
                doc_valid_mask=doc_valid_mask,
                doc_attention_mask=doc_attention_mask,
                doc_input_ids=doc_input_ids,
                comp_ratio_or_len_override=comp_ratio_or_len_override,
            )
            return inputs_embeds, attention_mask, None, None, None, predicted_compress_ratio, None

        # if comp_ratio_or_len_override is None, that means the dynamic-ratio mode, so we use the dynamic-ratio replace_placeholder_tokens
        num_valid_tokens = doc_valid_mask.sum(dim=1)
        inputs_embeds, new_attention_mask, placeholder_pos, new_len_list, pad_on_right = self.replace_placeholder_tokens_expand_one(
            input_ids=input_ids,
            compressed_doc_features=compressed_doc_features,
            doc_valid_mask=doc_valid_mask,
            attention_mask=attention_mask,
        )
        return inputs_embeds, new_attention_mask, placeholder_pos, num_valid_tokens, new_len_list, predicted_compress_ratio, pad_on_right

    def forward(
        self,
        doc_input_ids: torch.Tensor,
        input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compress_len_labels: Optional[torch.Tensor] = None,
        comp_ratio_or_len_override: Optional[Union[int, float]] = None,
        compress_ratio_scale: float = 0,
    ):
        """
        Forward with optional compression ratio loss. When comp_ratio_or_len_override is None, input_ids
        must contain exactly one placeholder per sample (expanded to M inside).

        Args:
            doc_input_ids, input_ids, doc_attention_mask, attention_mask, comp_ratio_or_len_override,
            compress_ratio_scale: Same as compute_inputs_embeds / get_doc_features.
            labels: Optional LM labels; when using single-placeholder expansion, labels are expanded with -100 at injected positions.
            compress_len_labels: Optional [batch_size] or [batch_size, 1] = log2(context_length/summary_length).
                If provided, comp_ratio_loss = MSE(predicted_compress_ratio, compress_len_labels) is added to the output loss.

        Returns:
            Decoder output with optional "lm_loss", "comp_ratio_loss", "num_valid_tokens", and combined "loss".
        """
        inputs_embeds, out_attention_mask, placeholder_pos, num_valid_tokens, new_len_list, predicted_compress_ratio, pad_on_right = self.compute_inputs_embeds(
            doc_input_ids=doc_input_ids,
            input_ids=input_ids,
            doc_attention_mask=doc_attention_mask,
            attention_mask=attention_mask,
            comp_ratio_or_len_override=comp_ratio_or_len_override,
            compress_ratio_scale=compress_ratio_scale,
        )
        if placeholder_pos is not None and labels is not None:
            max_len = inputs_embeds.shape[1]
            new_labels = self._expand_labels(
                labels, placeholder_pos, num_valid_tokens, new_len_list, max_len, pad_on_right,
            )
        else:
            new_labels = labels
        self._switch_decoder_adapter()
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=out_attention_mask,
            labels=new_labels,
        )
        lm_loss = getattr(outputs, "loss", None) if not isinstance(outputs, dict) else outputs.get("loss")
        comp_ratio_loss = None
        if compress_len_labels is not None and predicted_compress_ratio is not None:
            batch_size = predicted_compress_ratio.shape[0]
            target_log = compress_len_labels.view(batch_size, 1).to(
                predicted_compress_ratio.device
            ).to(predicted_compress_ratio.dtype).float()
            comp_ratio_loss = nn.functional.mse_loss(predicted_compress_ratio, target_log)
        loss = (lm_loss + comp_ratio_loss) if (lm_loss is not None and comp_ratio_loss is not None) else lm_loss
        if isinstance(outputs, dict):
            outputs = dict(outputs)
            outputs["lm_loss"] = lm_loss
            outputs["comp_ratio_loss"] = comp_ratio_loss
            outputs["num_valid_tokens"] = num_valid_tokens
            if loss is not None:
                outputs["loss"] = loss
        else:
            outputs.__dict__["lm_loss"] = lm_loss
            outputs.__dict__["comp_ratio_loss"] = comp_ratio_loss
            outputs.__dict__["num_valid_tokens"] = num_valid_tokens
            if loss is not None:
                outputs.__dict__["loss"] = loss
        return outputs

    def generate(
        self,
        doc_input_ids: torch.Tensor,
        input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        comp_ratio_or_len_override: Optional[Union[int, float]] = None,
        compress_ratio_scale: float = 0,
        *args,
        **kwargs,
    ):
        """
        Generate. When comp_ratio_or_len_override is None, each sample must have exactly one placeholder;
        M is predicted and that placeholder is expanded to M positions. *args and **kwargs are passed to decoder.generate.
        """
        with torch.no_grad():
            inputs_embeds, new_attention_mask, _, num_valid_tokens, _, _, _ = self.compute_inputs_embeds(
                doc_input_ids=doc_input_ids,
                input_ids=input_ids,
                doc_attention_mask=doc_attention_mask,
                attention_mask=attention_mask,
                comp_ratio_or_len_override=comp_ratio_or_len_override,
                compress_ratio_scale=compress_ratio_scale,
            )
            self._switch_decoder_adapter()
            generation_outputs = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask,
                *args,
                **kwargs,
            )
        if hasattr(generation_outputs, "__dict__"):
            generation_outputs.__dict__["num_valid_tokens"] = num_valid_tokens
        return generation_outputs

    def save_projector(self, path: str) -> None:
        """Save projector and compress_ratio_head state dicts to a single file (e.g. projector.pth)."""
        state = {
            "projector": self.projector.state_dict(),
            "compress_ratio_head": self.compress_ratio_head.state_dict(),
        }
        torch.save(state, path)

    def load_projector(self, path: str) -> None:
        """Load projector and compress_ratio_head from path. Compatible with checkpoint that has only projector."""
        try:
            state = torch.load(path, map_location=self.decoder.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=self.decoder.device)
        if isinstance(state, dict) and "projector" in state:
            self.projector.load_state_dict(state["projector"])
            if "compress_ratio_head" in state:
                self.compress_ratio_head.load_state_dict(state["compress_ratio_head"])
        else:
            self.projector.load_state_dict(state)

    def _get_save_config(self) -> dict:
        config = super()._get_save_config()
        config["discretize_ratio_mode"] = self.discretize_ratio_mode
        config["discretize_compare_in_log"] = self.discretize_compare_in_log
        return config

    @classmethod
    def from_pretrained(cls, path: str, share_base_model_inference: bool = False, **kwargs):
        """
        Load CtxCompSemiDynamicModel from directory. Loads base components via CtxCompModel.from_pretrained,
        then builds CtxCompSemiDynamicModel and loads projector (including compress_ratio_head). Config must
        contain discretize_ratio_mode and discretize_compare_in_log (or defaults are used).
        """
        base_model = CtxCompModel.from_pretrained(path, share_base_model_inference=share_base_model_inference, **kwargs)
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(
            encoder=base_model.encoder,
            decoder=base_model.decoder,
            placeholder_token_id=base_model.placeholder_token_id,
            memory_token_begin_id=base_model.memory_token_begin_id,
            comp_ratio_or_len=base_model.comp_ratio_or_len,
            mlp_converter_hidden_dim=base_model.projector.intermediate_size,
            feature_extract_method=base_model.feature_extract_method,
            encoder_training=base_model.encoder_training,
            decoder_training=base_model.decoder_training,
            discretize_ratio_mode=config.get("discretize_ratio_mode", "round"),
            discretize_compare_in_log=config.get("discretize_compare_in_log", False),
        )
        model._shared_base_model = base_model._shared_base_model
        model.load_projector(os.path.join(path, "projector.pth"))
        return model

    def load_from_checkpoint(self, path: str, **kwargs) -> None:
        super().load_from_checkpoint(path, **kwargs)
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        self.discretize_ratio_mode = config.get("discretize_ratio_mode", "round")
        self.discretize_compare_in_log = config.get("discretize_compare_in_log", False)

