# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model."""
import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.bart.configuration_bart import BartConfig
from torch.optim import AdamW

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "BartConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"
_SEQ_CLASS_EXPECTED_LOSS = 0.0
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"
_QA_EXPECTED_LOSS = 0.59
_QA_EXPECTED_OUTPUT = "' nice puppet'"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # see all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_causal: bool = False,
            config: Optional[BartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


BART_ATTENTION_CLASSES = {
    "eager": BartAttention,
    # "sdpa": BartSdpaAttention,
    # "flash_attention_2": BartFlashAttention2,
}


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
        self.self_attn = BART_ATTENTION_CLASSES["eager"](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            layer_head_mask: torch.FloatTensor,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
        self.self_attn = BART_ATTENTION_CLASSES["eager"](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
        self.encoder_attn = BART_ATTENTION_CLASSES["eager"](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartPreTrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"BartEncoderLayer", r"BartDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPreTrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


class BartPretrainedModel(BartPreTrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    >>> ARTICLE_TO_SUMMARIZE = (
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
    ```

    Mask filling example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    >>> TXT = "My friends are <mask> but they eat too many carbs."
    >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    >>> logits = model(input_ids).logits

    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    >>> probs = logits[0, masked_index].softmax(dim=0)
    >>> values, predictions = probs.topk(5)

    >>> tokenizer.decode(predictions).split()
    ['not', 'good', 'healthy', 'great', 'very']
    ```
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.013461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class BartEncoder(BartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._use_sdpa and head_mask is None and not output_attentions:
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # self.layer_norm = nn.LayerNorm(config.d_model)
        # 我要 36 个参数 a1 到 a36
        self.a1 = nn.Parameter(torch.tensor(0.01))
        self.a2 = nn.Parameter(torch.tensor(0.01))
        self.a3 = nn.Parameter(torch.tensor(0.01))
        self.a4 = nn.Parameter(torch.tensor(0.01))
        self.a5 = nn.Parameter(torch.tensor(0.01))
        self.a6 = nn.Parameter(torch.tensor(0.01))
        self.a7 = nn.Parameter(torch.tensor(0.01))
        self.a8 = nn.Parameter(torch.tensor(0.01))
        self.a9 = nn.Parameter(torch.tensor(0.01))
        self.a10 = nn.Parameter(torch.tensor(0.01))
        self.a11 = nn.Parameter(torch.tensor(0.01))
        self.a12 = nn.Parameter(torch.tensor(0.01))
        self.b1 = nn.Parameter(torch.tensor(0.01))
        self.b2 = nn.Parameter(torch.tensor(0.01))
        self.b3 = nn.Parameter(torch.tensor(0.01))
        self.b4 = nn.Parameter(torch.tensor(0.01))
        self.b5 = nn.Parameter(torch.tensor(0.01))
        self.b6 = nn.Parameter(torch.tensor(0.01))
        self.b7 = nn.Parameter(torch.tensor(0.01))
        self.b8 = nn.Parameter(torch.tensor(0.01))
        self.b9 = nn.Parameter(torch.tensor(0.01))
        self.b10 = nn.Parameter(torch.tensor(0.01))
        self.b11 = nn.Parameter(torch.tensor(0.01))
        self.b12 = nn.Parameter(torch.tensor(0.01))
        self.c1 = nn.Parameter(torch.tensor(0.01))
        self.c2 = nn.Parameter(torch.tensor(0.01))
        self.c3 = nn.Parameter(torch.tensor(0.01))
        self.c4 = nn.Parameter(torch.tensor(0.01))
        self.c5 = nn.Parameter(torch.tensor(0.01))
        self.c6 = nn.Parameter(torch.tensor(0.01))
        self.c7 = nn.Parameter(torch.tensor(0.01))
        self.c8 = nn.Parameter(torch.tensor(0.01))
        self.c9 = nn.Parameter(torch.tensor(0.01))
        self.c10 = nn.Parameter(torch.tensor(0.01))
        self.c11 = nn.Parameter(torch.tensor(0.01))
        self.c12 = nn.Parameter(torch.tensor(0.01))

        # self.a1 = 0.01
        # self.a2 = 0.01
        # self.a3 = 0.01
        # self.a4 = 0.01
        # self.a5 = 0.01
        # self.a6 = 0.01
        # self.a7 = 0.01
        # self.a8 = 0.01
        # self.a9 = 0.01
        # self.a10 = 0.01
        # self.a11 = 0.01
        # self.a12 = 0.01

        # self.b1 = 0.01
        # self.b2 = 0.01
        # self.b3 = 0.01
        # self.b4 = 0.01
        # self.b5 = 0.01
        # self.b6 = 0.01
        # self.b7 = 0.01
        # self.b8 = 0.01
        # self.b9 = 0.01
        # self.b10 = 0.01
        # self.b11 = 0.01
        # self.b12 = 0.01


        # self.c1 = 0.01
        # self.c2 = 0.01
        # self.c3 = 0.01
        # self.c4 = 0.01
        # self.c5 = 0.01
        # self.c6 = 0.01
        # self.c7 = 0.01
        # self.c8 = 0.01
        # self.c9 = 0.01
        # self.c10 = 0.01
        # self.c11 = 0.01
        # self.c12 = 0.01

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            topic_hidden_states: Optional[torch.FloatTensor] = None,
            topic_attention_mask: Optional[torch.LongTensor] = None,
            sent_hidden_states: Optional[torch.FloatTensor] = None,
            sent_attention_mask: Optional[torch.LongTensor] = None,
            tri_hidden_states: Optional[torch.FloatTensor] = None,
            tri_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions and cross_attn_head_mask is None:
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                if idx == 0:
                    new_encoder_hidden_states = encoder_hidden_states + self.a1 * topic_hidden_states + self.b1 * sent_hidden_states + self.c1 * tri_hidden_states
                elif idx == 1:
                    new_encoder_hidden_states = encoder_hidden_states + self.a2 * topic_hidden_states + self.b2 * sent_hidden_states + self.c2 * tri_hidden_states
                elif idx == 2:
                    new_encoder_hidden_states = encoder_hidden_states + self.a3 * topic_hidden_states + self.b3 * sent_hidden_states + self.c3 * tri_hidden_states
                elif idx == 3:
                    new_encoder_hidden_states = encoder_hidden_states + self.a4 * topic_hidden_states + self.b4 * sent_hidden_states + self.c4 * tri_hidden_states
                elif idx == 4:
                    new_encoder_hidden_states = encoder_hidden_states + self.a5 * topic_hidden_states + self.b5 * sent_hidden_states + self.c5 * tri_hidden_states
                elif idx == 5:
                    new_encoder_hidden_states = encoder_hidden_states + self.a6 * topic_hidden_states + self.b6 * sent_hidden_states + self.c6 * tri_hidden_states
                elif idx == 6:
                    new_encoder_hidden_states = encoder_hidden_states + self.a7 * topic_hidden_states + self.b7 * sent_hidden_states + self.c7 * tri_hidden_states
                elif idx == 7:
                    new_encoder_hidden_states = encoder_hidden_states + self.a8 * topic_hidden_states + self.b8 * sent_hidden_states + self.c8 * tri_hidden_states
                elif idx == 8:
                    new_encoder_hidden_states = encoder_hidden_states + self.a9 * topic_hidden_states + self.b9 * sent_hidden_states + self.c9 * tri_hidden_states
                elif idx == 9:
                    new_encoder_hidden_states = encoder_hidden_states + self.a10 * topic_hidden_states + self.b10 * sent_hidden_states + self.c10 * tri_hidden_states
                elif idx == 10:
                    new_encoder_hidden_states = encoder_hidden_states + self.a11 * topic_hidden_states + self.b11 * sent_hidden_states + self.c11 * tri_hidden_states
                elif idx == 11:
                    new_encoder_hidden_states = encoder_hidden_states + self.a12 * topic_hidden_states + self.b12 * sent_hidden_states + self.c12 * tri_hidden_states
                # new_encoder_hidden_states = self.layer_norm(new_encoder_hidden_states)
                # new_encoder_hidden_states = encoder_hidden_states
                # print("encoder_attention_mask.shape:", encoder_attention_mask.shape, "encoder_hidden_states.shape:", encoder_hidden_states.shape, "topic_attention_mask.shape:", topic_attention_mask.shape, "sent_attention_mask.shape:", sent_attention_mask.shape, "tri_attention_mask.shape:", tri_attention_mask.shape)
                new_encoder_attention_mask = encoder_attention_mask
                # new_encoder_attention_mask = torch.cat([new_encoder_attention_mask, sent_attention_mask], dim=1)
                # new_encoder_attention_mask = torch.cat([new_encoder_attention_mask, tri_attention_mask], dim=1)

                # expand encoder attention mask
                if encoder_hidden_states is not None and new_encoder_attention_mask is not None:
                    if self._use_flash_attention_2:
                        new_encoder_attention_mask = new_encoder_attention_mask if 0 in new_encoder_attention_mask else None
                    elif self._use_sdpa and cross_attn_head_mask is None and not output_attentions:
                        # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                        # the manual implementation that requires a 4D causal mask in all cases.
                        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                        new_encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                            new_encoder_attention_mask,
                            inputs_embeds.dtype,
                            tgt_len=input_shape[-1],
                        )
                    else:
                        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                        new_encoder_attention_mask = _prepare_4d_attention_mask(
                            new_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                        )

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=new_encoder_hidden_states,
                    encoder_attention_mask=new_encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class CosineSimilarityNet(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(CosineSimilarityNet, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # 添加可学习的线性层
        # self.topic_transform = nn.Linear(d_model, hidden_dim)
        # self.document_transform = nn.Linear(d_model, hidden_dim)

    def forward(self, topics_hidden_state, document_hidden_state):
        # 对输入进行线性变换
        # transformed_topics = self.topic_transform(topics_hidden_state)
        # transformed_documents = self.document_transform(document_hidden_state)

        # 对序列维度进行平均池化，以获得固定长度的向量表示
        transformed_topics = torch.mean(topics_hidden_state, dim=1)
        transformed_documents = torch.mean(document_hidden_state, dim=1)

        # 归一化处理
        transformed_topics_norm = transformed_topics / torch.norm(transformed_topics, dim=-1, keepdim=True)
        transformed_documents_norm = transformed_documents / torch.norm(transformed_documents, dim=-1, keepdim=True)

        # 计算余弦相似度
        cosine_similarity = (transformed_topics_norm * transformed_documents_norm).sum(dim=-1)

        # 计算损失：我们希望余弦相似度接近1
        loss = 1 - cosine_similarity.mean()
        return loss


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModel(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.embed_dim = config.d_model
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.dropout = 0.01

        # --------------------- Topic Module -----------------------------
        self.topic_embedding = self.shared
        # self.topic_Self_Attn = BartEncoderLayer(config)
        self.topic_embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.topic_layernorm_embedding = nn.LayerNorm(config.d_model)
        self.topic_Cross_Attn_1 = BartDecoderLayer(config)
        # -------------------------------------------------------

        # --------------------- Sentence Module -----------------------------
        self.sent_embedding = self.shared
        # self.sent_Self_Attn = BartEncoderLayer(config)
        self.sent_embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.sent_layernorm_embedding = nn.LayerNorm(config.d_model)
        self.sent_Cross_Attn_1 = BartDecoderLayer(config)
        # self.sent_Cross_Attn_2 = BartDecoderLayer(config)
        # -------------------------------------------------------

        # --------------------- Triple Module -----------------------------
        self.tri_embedding = self.shared
        # self.tri_Self_Attn = BartEncoderLayer(config)
        self.tri_embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.tri_layernorm_embedding = nn.LayerNorm(config.d_model)
        self.tri_Cross_Attn_1 = BartDecoderLayer(config)
        # self.tri_Cross_Attn_2 = BartDecoderLayer(config)
        # self.tri_Cross_Attn_3 = BartDecoderLayer(config)
        # -------------------------------------------------------

        self.cosine_similarity_net_1 = CosineSimilarityNet(config.d_model, 128)
        self.cosine_similarity_net_2 = CosineSimilarityNet(config.d_model, 128)
        self.cosine_similarity_net_3 = CosineSimilarityNet(config.d_model, 128)
        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            topic_labels: Optional[torch.LongTensor] = None,
            topic_attention_mask: Optional[torch.Tensor] = None,

            sent_labels: Optional[torch.LongTensor] = None,
            sent_attention_mask: Optional[torch.Tensor] = None,

            tri_labels: Optional[torch.LongTensor] = None,
            tri_attention_mask: Optional[torch.Tensor] = None,

            other_input_ids: Optional[torch.LongTensor] = None,
            other_attention_mask: Optional[torch.Tensor] = None,


    ) -> Union[Tuple, Seq2SeqModelOutput]:
        #############################################################################################
        # topic_decoder_input_ids = shift_tokens_right(
        #     topic_labels, self.config.pad_token_id, self.config.decoder_start_token_id
        # )
        if other_input_ids is not None:
            input_ids = other_input_ids
            attention_mask = other_attention_mask

        topic_decoder_inputs_embeds = self.topic_embedding(input_ids)
        positions = self.topic_embed_positions(input_ids)
        positions = positions.to(input_ids.device)
        topic_hidden_states = topic_decoder_inputs_embeds + positions
        topic_hidden_states = self.topic_layernorm_embedding(topic_hidden_states)
        topic_hidden_states = nn.functional.dropout(topic_hidden_states, p=self.dropout, training=self.training)
        expend_topic_attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask,
                                                                          topic_decoder_inputs_embeds.dtype)
        # print("topic_hidden_states:", topic_hidden_states)
        # topic_hidden_states = self.topic_Self_Attn(topic_hidden_states, attention_mask=expend_topic_attention_mask,
        #                                            layer_head_mask=None)
        # print("topic_hidden_states.type:", type(topic_hidden_states))
        # print("t_topic_hidden_states.type:", type(t_topic_hidden_states))
        # print("topic_hidden_states.shape:", topic_hidden_states.shape)
        # print("t_topic_hidden_states[0].shape:", t_topic_hidden_states[0].shape)
        # exit(1)
        # sent_decoder_input_ids = shift_tokens_right(
        #     sent_labels, self.config.pad_token_id, self.config.decoder_start_token_id
        # )
        sent_decoder_inputs_embeds = self.sent_embedding(input_ids)
        positions = self.sent_embed_positions(input_ids)
        positions = positions.to(sent_decoder_inputs_embeds.device)
        sent_hidden_states = sent_decoder_inputs_embeds + positions
        sent_hidden_states = self.sent_layernorm_embedding(sent_hidden_states)
        sent_hidden_states = nn.functional.dropout(sent_hidden_states, p=self.dropout, training=self.training)
        expend_sent_attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask,
                                                                         sent_decoder_inputs_embeds.dtype)
        # sent_hidden_states = self.sent_Self_Attn(sent_hidden_states, attention_mask=expend_sent_attention_mask,
        #                                          layer_head_mask=None)

        # tri_decoder_input_ids = shift_tokens_right(
        #     tri_labels, self.config.pad_token_id, self.config.decoder_start_token_id
        # )

        tri_decoder_inputs_embeds = self.tri_embedding(input_ids)
        positions = self.tri_embed_positions(input_ids)
        positions = positions.to(tri_decoder_inputs_embeds.device)
        tri_hidden_states = tri_decoder_inputs_embeds + positions
        tri_hidden_states = self.tri_layernorm_embedding(tri_hidden_states)
        tri_hidden_states = nn.functional.dropout(tri_hidden_states, p=self.dropout, training=self.training)
        expend_tri_attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, tri_decoder_inputs_embeds.dtype)
        # tri_hidden_states = self.tri_Self_Attn(tri_hidden_states, attention_mask=expend_tri_attention_mask,
        #                                        layer_head_mask=None)
        #############################################################################################
        if topic_labels is not None:
            with torch.no_grad():
                topic_outputs = self.encoder(
                    input_ids=topic_labels,
                    attention_mask=topic_attention_mask
                )
                topic_label_hidden_states = topic_outputs[0]
                sent_outputs = self.encoder(
                    input_ids=sent_labels,
                    attention_mask=sent_attention_mask
                )
                sent_label_hidden_states = sent_outputs[0]
                tri_outputs = self.encoder(
                    input_ids=tri_labels,
                    attention_mask=tri_attention_mask
                )
                tri_label_hidden_states = tri_outputs[0]
        else:
            topic_label_hidden_states = None
            sent_label_hidden_states = None
            tri_label_hidden_states = None


        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            with torch.no_grad():
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        ######################## first cross attention (start) ###################################
        # expend_topic_attention_mask = _prepare_4d_attention_mask_for_sdpa(topic_attention_mask, tri_decoder_inputs_embeds.dtype)

        expend_encoder_topic_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask,
            topic_decoder_inputs_embeds.dtype,
            tgt_len=input_ids.shape[-1],
        )

        topic_hidden_states = self.topic_Cross_Attn_1(
            topic_hidden_states,
            attention_mask=expend_topic_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=expend_encoder_topic_attention_mask,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            # past_key_value=past_key_value,  # training is None
            output_attentions=output_attentions,
            use_cache=use_cache,  # training is None
        )
        ######################## first cross attention (end) ###################################

        ######################## second cross attention (start) ###################################
        expend_encoder_sent_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask,
            sent_decoder_inputs_embeds.dtype,
            tgt_len=input_ids.shape[-1],
        )

        sent_hidden_states = self.sent_Cross_Attn_1(
            sent_hidden_states,
            attention_mask=expend_sent_attention_mask,
            encoder_hidden_states=topic_hidden_states[0],
            encoder_attention_mask=expend_encoder_sent_attention_mask,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            # past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        ######################## second cross attention (end) ############################

        ####################### trird cross attention (start) ###############
        expend_encoder_tri_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask,
            sent_decoder_inputs_embeds.dtype,
            tgt_len=input_ids.shape[-1],
        )

        tri_hidden_states = self.tri_Cross_Attn_1(
            tri_hidden_states,
            attention_mask=expend_tri_attention_mask,
            encoder_hidden_states=sent_hidden_states[0],
            encoder_attention_mask=expend_encoder_tri_attention_mask,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            # past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        ####################### trird cross attention (end) ###############

        if topic_label_hidden_states is not None:
            loss_1 = self.cosine_similarity_net_1(topic_label_hidden_states, encoder_outputs[0])
            loss_2 = self.cosine_similarity_net_2(sent_label_hidden_states, encoder_outputs[0])
            loss_3 = self.cosine_similarity_net_3(tri_label_hidden_states, encoder_outputs[0])
        else:
            loss_1 = 0
            loss_2 = 0
            loss_3 = 0

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            topic_hidden_states=topic_hidden_states[0],
            topic_attention_mask=attention_mask,
            sent_hidden_states=sent_hidden_states[0],
            sent_attention_mask=attention_mask,
            tri_hidden_states=tri_hidden_states[0],
            tri_attention_mask=attention_mask,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), loss_1, loss_2, loss_3


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class CombinationModel(BartPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BartConfig, model_state_dict=None):
        super().__init__(config)
        config.vocab_size = 50264
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def generate_t(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            max_time: Optional[float] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            diversity_penalty: Optional[float] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            **model_kwargs,
    ):
        return self.generate(input_ids=input_ids,
                             max_length=max_length,
                             min_length=min_length,
                             do_sample=do_sample,
                             early_stopping=early_stopping,
                             num_beams=num_beams,
                             temperature=temperature,
                             top_k=top_k,
                             top_p=top_p,
                             repetition_penalty=repetition_penalty,
                             bos_token_id=bos_token_id,
                             pad_token_id=pad_token_id,
                             eos_token_id=eos_token_id,
                             length_penalty=length_penalty,
                             no_repeat_ngram_size=no_repeat_ngram_size,
                             encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                             max_time=max_time,
                             decoder_start_token_id=decoder_start_token_id,
                             use_cache=use_cache,
                             diversity_penalty=diversity_penalty,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             output_scores=output_scores,
                             return_dict_in_generate=return_dict_in_generate,
                             forced_bos_token_id=forced_bos_token_id,
                             forced_eos_token_id=forced_eos_token_id,
                             remove_invalid_values=remove_invalid_values,
                             synced_gpus=synced_gpus,
                             **model_kwargs)

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            doc_labels: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            topic_labels: Optional[torch.LongTensor] = None,
            topic_attention_mask: Optional[torch.Tensor] = None,

            sent_labels: Optional[torch.LongTensor] = None,
            sent_attention_mask: Optional[torch.Tensor] = None,

            tri_labels: Optional[torch.LongTensor] = None,
            tri_attention_mask: Optional[torch.Tensor] = None,

            other_input_ids: Optional[torch.LongTensor] = None,
            other_attention_mask: Optional[torch.Tensor] = None,

    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if doc_labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    doc_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs, topic_loss, sent_loss, tri_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            topic_labels=topic_labels,
            topic_attention_mask=topic_attention_mask,

            sent_labels=sent_labels,
            sent_attention_mask=sent_attention_mask,

            tri_labels=tri_labels,
            tri_attention_mask=tri_attention_mask,

            other_input_ids=other_input_ids,
            other_attention_mask=other_attention_mask,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if doc_labels is not None:
            doc_labels = doc_labels.to(lm_logits.device)
            loss_fct_1 = CrossEntropyLoss()
            masked_lm_loss = loss_fct_1(lm_logits.view(-1, self.config.vocab_size), doc_labels.view(-1))

        if masked_lm_loss is None:
            loss = None
        else:
            loss = 0.1 * masked_lm_loss +  0.9 * (topic_loss +  sent_loss +  tri_loss)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "other_input_ids": kwargs["other_input_ids"],
            "other_attention_mask": kwargs["other_attention_mask"]
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


if __name__ == '__main__':
    # determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # arguments
    batch_size = 4
    seq_len = 200
    d_model = 1024  # hidden_state size
    vocab_size = 50264


    def train(doc_x, doc_y, topic_y, sent_y, tri_y, model, optim):
        # model returns the prediction and the loss that encourages all experts to have equal importance and load
        model.train()
        outputs = model(input_ids=doc_x['input_ids'],
                        attention_mask=doc_x['attention_mask'],
                        doc_labels=doc_y['input_ids'],
                        decoder_attention_mask=doc_y['attention_mask'],

                        topic_labels=topic_y['input_ids'],
                        topic_attention_mask=topic_y['attention_mask'],

                        sent_labels=sent_y['input_ids'],
                        sent_attention_mask=sent_y['attention_mask'],

                        tri_labels=tri_y['input_ids'],
                        tri_attention_mask=tri_y['attention_mask'],
                        )
        loss = outputs.loss
        logits = outputs.logits
        optim.zero_grad()
        loss.backward()
        optim.step()
        print("Training Results - loss: {:.2f}, logits.shape: {}".format(loss.item(), logits.shape))
        return model


    def eval(doc_x, doc_y, topic_y, sent_y, tri_y, model):
        model.eval()
        outputs = model(input_ids=doc_x['input_ids'],
                        attention_mask=doc_x['attention_mask'],
                        doc_labels=doc_y['input_ids'],
                        decoder_attention_mask=doc_y['attention_mask'],

                        topic_labels=topic_y['input_ids'],
                        topic_attention_mask=topic_y['attention_mask'],

                        sent_labels=sent_y['input_ids'],
                        sent_attention_mask=sent_y['attention_mask'],

                        tri_labels=tri_y['input_ids'],
                        tri_attention_mask=tri_y['attention_mask'],
                        )
        loss = outputs.loss
        logits = outputs.loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        print("Eval Results - loss: {:.2f}, logits.shape: {}".format(loss.item(), logits.shape))
        return model


    def dummy_data(batch_size, seq_len, vocab_size, device):
        x_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        x_attn_mask = torch.ones_like(x_ids)

        for i in range(len(x_attn_mask)):
            x_attn_mask[i, seq_len - (i + 1) * 2:] = 0

        y_ids = torch.randint(0, vocab_size, (batch_size, seq_len // 2))
        y_attn_mask = torch.ones_like(y_ids)
        for i in range(len(y_attn_mask)):
            y_attn_mask[i, seq_len // 10 - (i + 1):] = 0

        doc_x = {'input_ids': x_ids.to(device), 'attention_mask': x_attn_mask.to(device)}
        doc_y = {'input_ids': y_ids.to(device), 'attention_mask': y_attn_mask.to(device)}

        topic_y = {'input_ids': y_ids.to(device), 'attention_mask': y_attn_mask.to(device)}
        sent_y = {'input_ids': y_ids.to(device), 'attention_mask': y_attn_mask.to(device)}
        tri_y = {'input_ids': y_ids.to(device), 'attention_mask': y_attn_mask.to(device)}

        return doc_x, doc_y, topic_y, sent_y, tri_y


    c_model = CombinationModel(BartConfig())
    print("c_model:", c_model)

    total_model_params = sum(p.numel() for p in c_model.parameters())
    print(f"Total number of model parameters: {total_model_params}")
    params_memory = total_model_params * 4 / (1024 ** 2)
    print(f"Total memory required for model parameters: {params_memory} MB")

    c_model = c_model.to(device)
    doc_x, doc_y, topic_y, sent_y, tri_y = dummy_data(batch_size, seq_len, vocab_size, device)
    optim = AdamW(c_model.parameters(), lr=5e-4)
    # train
    train_model = train(doc_x, doc_y, topic_y, sent_y, tri_y, c_model, optim)
    # evaluate
    doc_x, doc_y, topic_y, sent_y, tri_y = dummy_data(batch_size, seq_len, vocab_size, device)
    eval(doc_x, doc_y, topic_y, sent_y, tri_y, train_model)
