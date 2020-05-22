import numpy as np
import torch
from tqdm import tqdm
from data.dataset import *


def load_spacy_glove_embedding(spacy_nlp, vocab):
    vocab_size = len(vocab.token2id)
    # print('vocab_size in function: ', vocab_size)
    word_vec_size = spacy_nlp.vocab.vectors_length
    embedding = np.zeros((vocab_size, word_vec_size))
    unk_count = 0

    print('=' * 100)
    print('Loading spacy glove embedding:')
    print('- Vocabulary size: {}'.format(vocab_size))
    print('- Word vector size: {}'.format(word_vec_size))
    for token, index in tqdm(vocab.token2id.items()):
        if token == vocab.id2token[PAD]:
            continue
        elif token in [vocab.id2token[BOS], vocab.id2token[EOS], vocab.id2token[UNK], vocab.id2token[BEGIN]]:
            vector = np.random.rand(word_vec_size, )
        elif spacy_nlp.vocab[token].has_vector:
            vector = spacy_nlp.vocab[token].vector
        else:
            # print('UNK token: ', token)
            vector = embedding[UNK]
            unk_count += 1

        embedding[index] = vector

    print('- Unknown word count: {}'.format(unk_count))
    print('=' * 100 + '\n')
    return torch.from_numpy(embedding).float()
