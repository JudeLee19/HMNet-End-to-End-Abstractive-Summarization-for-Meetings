from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import math
from .sublayers import MultiHeadAttention, PositionwiseFeedForward
from ..normalization import LayerNorm
from collections import defaultdict
import time


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1) # [1, num_heads, max_length, max_length]


def _gen_seq_bias_mask(valid_length_list, max_seq_length):
    seq_mask = np.full([len(valid_length_list), max_seq_length, max_seq_length], 0)
    for idx, length in enumerate(valid_length_list):
        seq_mask[idx, :, length:] = -np.inf

    for idx, length in enumerate(valid_length_list):
        seq_mask[idx, length:] = -np.inf

    seq_mask = torch.from_numpy(seq_mask).type(torch.FloatTensor) # [num_turns, max_seq_length, max_seq_length]

    return seq_mask.unsqueeze(1) # [num_turns, 1, max_seq_length, max_seq_length]


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])
    return torch.from_numpy(signal).type(torch.FloatTensor)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, src_masks=None):

        x, src_masks = inputs, src_masks

        # Layer Norm
        x_norm = self.layer_norm_mha(x)

        # Multi-head
        y = self.multi_head_attention(x_norm, x_norm, x_norm, src_masks)

        # Dropout & Residual
        x = self.dropout(x + y)

        # Layer Norm
        x_norm = self.layer_norm_ffn(x)

        y = self.positionwise_feed_forward(x_norm)

        y = self.dropout(y + x)
        return y


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, use_pos=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        super(Encoder, self).__init__()
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        # Pos-tag & Entity feature should be added later.
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)

        self.encoder_layers = nn.Sequential(*[EncoderLayer(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, src_masks=None, role_inputs=None):

        # Construct Transformer-Encoder input representation. inputs is the result vectors of glove & pos embeddings.

        if role_inputs is not None:
            inputs = torch.cat((inputs, role_inputs), dim=-1)

        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # y = self.encoder_layers((x, src_masks))
        y = x
        for idx, encoder_layer in enumerate(self.encoder_layers):
            y = encoder_layer(inputs=y, src_masks=src_masks)

        y = self.layer_norm(y)
        return y


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask=bias_mask,
                                                           dropout=attention_dropout,
                                                           attention_type='self-attention')

        self.multi_head_attention_word = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, None, dropout=attention_dropout,
                                                            attention_type='word-attention')

        self.multi_head_attention_turn = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, None, dropout=attention_dropout,
                                                            attention_type='turn-attention')

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding = 'left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_word_enc = LayerNorm(hidden_size)
        self.layer_norm_mha_turn_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        self.bias_mask = bias_mask

    def forward(self, inputs, layer_cache=None):
        decoder_inputs, word_encoder_outputs, turn_encoder_outputs = inputs

        x_norm = self.layer_norm_mha_dec(decoder_inputs)

        # Masked Multi-head Self-attention for decoding inputs
        start_time = time.time()
        y = self.multi_head_attention_dec(x_norm, x_norm,
                                          x_norm, layer_cache=layer_cache)
        # print('[Self-Attention]: ', int(round((time.time() - start_time) * 1000)), 'MS')


        x = self.dropout(decoder_inputs + y) # [1, tgt_seq_len, 300]

        # Layer Normalization for word-level attention
        x_norm = self.layer_norm_mha_word_enc(x)


        start_time = time.time()
        # Word-level cross-attention
        y = self.multi_head_attention_word(x_norm, word_encoder_outputs,
                                           word_encoder_outputs,
                                           layer_cache=layer_cache)
        # print('[Word-level cross-attention]: ', int(round((time.time() - start_time) * 1000)), 'MS')

        x = self.dropout(x + y)

        # Layer Norm of turn-level cross attention
        x_norm = self.layer_norm_mha_turn_enc(x)

        start_time = time.time()
        # Turn-level cross-attention
        y = self.multi_head_attention_turn(x_norm, turn_encoder_outputs,
                                           turn_encoder_outputs,
                                           layer_cache=layer_cache)
        # print('[Turn-level cross-attention]: ', int(round((time.time() - start_time) * 1000)), 'MS')

        x = self.dropout(x + y)

        # Layer Norm of turn-level cross attention
        x_norm = self.layer_norm_ffn(x)

        # Position-wise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)
        return y, word_encoder_outputs, turn_encoder_outputs


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size) # [1, max_len, 300]

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.num_layers = num_layers
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.decoder_layers = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, state=None, step=None):
        decoder_inputs, word_encoder_outputs, turn_encoder_outputs = inputs

        # print('decoder_inputs: ', decoder_inputs)

        # Add input dropout
        x = self.input_dropout(decoder_inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal (training)
        if step is None:
            x += self.timing_signal[:, :decoder_inputs.shape[1], :].type_as(decoder_inputs.data)
        else:
            x += self.timing_signal[:, step, :].type_as(decoder_inputs.data)

        output = x
        # Run decoder
        if state is None:
            # y = x
            output, word_encoder_outputs, turn_encoder_outputs = self.decoder_layers((output, word_encoder_outputs, turn_encoder_outputs))
        else:
            # y = x
            # utilize state caching only for inference
            for idx, decoder_layer in enumerate(self.decoder_layers):
                output, word_encoder_outputs, turn_encoder_outputs = decoder_layer(inputs=(output, word_encoder_outputs, turn_encoder_outputs),
                                                                                   layer_cache=state.cache[
                                                                                       "layer_{}".format(idx)]
                                                                                   if state.cache is not None else None)

        # Final layer normalization
        y = self.layer_norm(output)

        return y, state

    def init_decoder_state(self):
        state = DecoderState()
        state._init_cache(self.num_layers)
        return state


class DecoderState(object):
    def __init__(self):
        self.previous_input = None
        self.previous_layer_inputs = None
        self.layer_caches = defaultdict(lambda: {'self-attention': None,
                                                 'word-attention': None,
                                                 'turn-attention': None})

        self.cache = None

    def _init_cache(self, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "word_keys": None,
                "word_values": None,
                "turn_keys": None,
                "turn_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        if self.cache is not None:
            _recursive_map(self.cache)