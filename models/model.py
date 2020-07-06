import spacy
from utils.utils import load_spacy_glove_embedding
import torch
import torch.nn as nn
from models import transformer


class SummarizationModel(nn.Module):
    def __init__(self, hparams=None, vocab_word=None, vocab_role=None, vocab_pos=None, checkpoint=None):
        super(SummarizationModel, self).__init__()
        self.hparams = hparams

        # Define Embedding layers
        self.vocab_word = vocab_word
        if self.vocab_word is None:
            raise ValueError('Must provide vocab_word !')

        self.vocab_role = vocab_role
        if hparams.use_role and self.vocab_role is None:
            raise ValueError('Must provide vocab_role !')

        self.vocab_pos = vocab_pos
        if hparams.use_pos and self.vocab_role is None:
            raise ValueError('Must provide vocab_role !')

        self.vocab_size = len(self.vocab_word.token2id)
        self.vocab_role_size = len(self.vocab_role.token2id)
        self.vocab_pos_size = len(self.vocab_pos.token2id)
        self.embedding_word = nn.Embedding(self.vocab_size, hparams.embedding_size_word)
        if checkpoint is None:
            # Load glove embeddings from spacy library
            nlp = spacy.load('en_core_web_lg')
            glove_embedding = load_spacy_glove_embedding(nlp, self.vocab_word)
            self.embedding_word.weight.data.copy_(glove_embedding)
            self.embedding_word.weight.requires_grad = hparams.fintune_word_embedding

        if self.hparams.use_pos:
            self.embedding_pos = nn.Embedding(self.vocab_pos_size, hparams.embedding_size_pos)

        if self.hparams.use_role:
            self.embedding_role = nn.Embedding(self.vocab_role_size, hparams.embedding_size_role)

        # Define word and turn-level Encoder
        self.word_level_encoder = transformer.Encoder(
            (hparams.embedding_size_word + hparams.embedding_size_pos) if self.hparams.use_pos else (hparams.embedding_size_word),
            hparams.hidden_size,
            hparams.num_hidden_layers,
            hparams.num_heads,
            hparams.attention_key_channels,
            hparams.attention_value_channels,
            hparams.filter_size,
            hparams.max_length,
            hparams.dropout,
            hparams.dropout,
            hparams.dropout,
            hparams.dropout,
            use_mask=False
        )

        self.turn_level_encoder = transformer.Encoder(
            (hparams.embedding_size_word + hparams.embedding_size_role) if self.hparams.use_role else (hparams.embedding_size_word),
            hparams.hidden_size, # 300
            hparams.num_hidden_layers,
            hparams.num_heads,
            hparams.attention_key_channels,
            hparams.attention_value_channels,
            hparams.filter_size,
            hparams.max_length,
            hparams.dropout,
            hparams.dropout,
            hparams.dropout,
            hparams.dropout,
            use_mask=False
        )

        # Define Decoder
        self.decoder = transformer.Decoder(
            hparams.embedding_size_word,
            hparams.hidden_size,
            hparams.num_hidden_layers,
            hparams.num_heads,
            hparams.attention_key_channels,
            hparams.attention_value_channels,
            hparams.filter_size,
            hparams.max_length,
            hparams.dropout,
            hparams.dropout,
            hparams.dropout,
            hparams.dropout,
            use_mask=True
        )

        # Reuse the weight of embedding matrix D, to decode v_{k-1} into a probability distribution
        self.final_linear = nn.Linear(self.embedding_word.embedding_dim, self.embedding_word.num_embeddings) # [300, vocab_size]
        if checkpoint is None:
            self.final_linear.weight = self.embedding_word.weight

    def forward(self, inputs, targets, src_masks=None, role_ids=None, pos_ids=None):
        """

        :param

        inputs: [batch_size, num_turns, padded_seq_len]
        targets: [batch_size, seq_len]
        src_mask: [num_turns, batch_size, padded_seq_len]

        :return:
        """

        src_masks = src_masks.squeeze(0) # [num_turns, batch_size, padded_seq_len, padded_seq_len]

        # Inputs Self-Attention
        inputs = torch.squeeze(inputs, 0) # [1, num_turns, seq_len]
        inputs_word_emb = self.embedding_word(inputs) # [num_turns, seq_len, word_dim==300]

        if self.hparams.use_pos:
            pos_ids = torch.squeeze(pos_ids, 0)
            inputs_pos_emb = self.embedding_pos(pos_ids) # [num_turns, seq_len, pos_dim==12]
            inputs_word_emb = torch.cat((inputs_word_emb, inputs_pos_emb), -1)

        # Word-level Attention
        word_level_outputs = self.word_level_encoder(inputs=inputs_word_emb,
                                                     src_masks=src_masks, role_inputs=None) # [num_turns, seq_len, 300]

        # Turn-level Attention
        turn_level_inputs = word_level_outputs[:, 0] # [num_turns, 300]
        turn_level_inputs = torch.unsqueeze(turn_level_inputs, 0) # [1, num_turns, 300]

        if self.hparams.use_role:
            role_ids = role_ids.squeeze(-1)
            turn_level_role_emb = self.embedding_role(role_ids) # [1, num_turns, role_dim==30]
            turn_level_outputs = self.turn_level_encoder(inputs=turn_level_inputs,
                                                         src_masks=None, role_inputs=turn_level_role_emb) # [1, num_turns, 300]
        else:
            turn_level_outputs = self.turn_level_encoder(inputs=turn_level_inputs,
                                                         src_masks=None,
                                                         role_inputs=None)  # [1, num_turns, 300]

        # Target Self-Attention
        targets_word_emb = self.embedding_word(targets) # [1, tgt_seq_len, 300]

        # word_level_outputs = word_level_outputs[:, 1:]
        word_level_shape = word_level_outputs.shape
        word_level_outputs = word_level_outputs.reshape(word_level_shape[0] * word_level_shape[1], 300)
        word_level_outputs = word_level_outputs.unsqueeze(0) # [1, num_turns x seq_len, 300]

        decoder_outputs, state = self.decoder((targets_word_emb, word_level_outputs, turn_level_outputs)) # [1, tgt_seq_len, 300]

        logits = self.final_linear(decoder_outputs)

        shape = logits.shape
        logits = logits.view(shape[0]*shape[1], shape[-1]) # [beam_size x tgt_seq_len, vocab_size]

        return logits


