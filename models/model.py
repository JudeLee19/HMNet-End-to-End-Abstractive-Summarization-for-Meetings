import spacy
from utils.utils import load_spacy_glove_embedding
import torch
import torch.nn as nn
from models import transformer


class SummarizationModel(nn.Module):
    def __init__(self, hparams=None, vocabs=None, checkpoint=None):
        super(SummarizationModel, self).__init__()
        self.hparams = hparams

        # Define Embedding layers
        self.vocab_word = vocabs
        if self.vocab_word is None:
            raise ValueError('Must provide vocab_word !')

        self.vocab_size = len(self.vocab_word.token2id)
        self.embedding_word = nn.Embedding(self.vocab_size, hparams.embedding_size_word)
        # Load glove embeddings from spacy library
        nlp = spacy.load('en_core_web_lg')
        glove_embedding = load_spacy_glove_embedding(nlp, self.vocab_word)
        self.embedding_word.weight.data.copy_(glove_embedding)
        self.embedding_word.weight.requires_grad = hparams.fintune_word_embedding

        # Define word and turn-level Encoder
        self.word_level_encoder = transformer.Encoder(
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
            use_mask=False
        )
        self.turn_level_encoder = transformer.Encoder(
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

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, src_masks):
        """

        :param

        inputs: [batch_size, num_turns, padded_seq_len]
        targets: [batch_size, seq_len]
        src_mask: [num_turns, batch_size, padded_seq_len]

        :return:
        """

        print('======= [In Summarization Model Forward function] =======')
        print('dialogues_ids shape: ', inputs.shape)
        print('targets shape: ', targets.shape)
        print('src_masks shape: ', src_masks.shape)
        print('=========================================================')

        inputs = torch.squeeze(inputs, 0) #[num_turns, seq_len]
        inputs_word_emb = self.embedding_word(inputs) # [num_turns, seq_len, 300]

        word_level_outputs = self.word_level_encoder(inputs=inputs_word_emb, src_masks=src_masks) # [num_turns, seq_len, 300]

        print('Word-level encoder is done')

        turn_level_inputs = word_level_outputs[:, 0] # [num_turns, 300]
        turn_level_inputs = torch.unsqueeze(turn_level_inputs, 0) # [1, num_turns, 300]
        turn_level_outputs = self.turn_level_encoder(turn_level_inputs) # [1, num_turns, 300]

        print('Turn-level encoder is done')

        # print('turn_level_outputs shape: ', turn_level_outputs.shape)

        # Target inputs
        targets_word_emb = self.embedding_word(targets) # [1, tgt_seq_len, 300]

        word_level_shape = word_level_outputs.shape
        word_level_outputs = word_level_outputs.view(word_level_shape[0] * word_level_shape[1], 300)
        word_level_outputs = word_level_outputs.unsqueeze(0)
        # print('word_level_outputs shape: ', word_level_outputs.shape)

        decoder_outputs = self.decoder((targets_word_emb, word_level_outputs, turn_level_outputs)) # [1, tgt_seq_len, 300]
        # print('decoder_outputs: ', decoder_outputs.shape)

        # Reuse the weight of embedding matrix D, to decode v_{k-1} into a probability distribution
        logitis = torch.matmul(decoder_outputs, self.embedding_word.transpose(0, 1))

        return logitis


