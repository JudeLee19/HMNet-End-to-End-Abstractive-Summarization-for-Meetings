import numpy as np
import torch
from tqdm import tqdm
from data.dataset import *
from rouge import Rouge


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


def compute_rouge_scores(cand_list, ref_list):
    """

    :param cand_list: list of candidate summaries
    :param ref_list: list of reference summaries
    :return: rouge scores
    """
    rouge = Rouge()
    rouge_1_f_score = 0.
    rouge_2_f_score = 0.
    rouge_L_f_score = 0.

    rouge_1_r_score = 0.
    rouge_2_r_score = 0.
    rouge_L_r_score = 0.

    rouge_1_p_score = 0.
    rouge_2_p_score = 0.
    rouge_L_p_score = 0.

    doc_count = len(cand_list)

    for cand, ref in zip(cand_list, ref_list):
        rouge_scores = rouge.get_scores(cand, ref)[0]
        rouge_1_f_score += rouge_scores['rouge-1']['f']
        rouge_2_f_score += rouge_scores['rouge-2']['f']
        rouge_L_f_score += rouge_scores['rouge-l']['f']

        rouge_1_r_score += rouge_scores['rouge-1']['r']
        rouge_2_r_score += rouge_scores['rouge-2']['r']
        rouge_L_r_score += rouge_scores['rouge-l']['r']

        rouge_1_p_score += rouge_scores['rouge-1']['p']
        rouge_2_p_score += rouge_scores['rouge-2']['p']
        rouge_L_p_score += rouge_scores['rouge-l']['p']
    rouge_1_f_score = rouge_1_f_score / doc_count
    rouge_2_f_score = rouge_2_f_score / doc_count
    rouge_L_f_score = rouge_L_f_score / doc_count

    results_dict = {}
    results_dict['rouge_1_f_score'] = rouge_1_f_score
    results_dict['rouge_2_f_score'] = rouge_2_f_score
    results_dict['rouge_l_f_score'] = rouge_L_f_score

    return results_dict


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x