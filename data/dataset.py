import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple
from tqdm import tqdm

PAD = 0
BOS = 1
EOS = 2
UNK = 3
BEGIN = 4


class AttrDict(dict):
    """ Access dictionary keys like attribute
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


class AMIDataset(Dataset):
    def __init__(self, hparams, type='', vocab_word=None, max_vocab_size=50000):
        super().__init__()
        self.hparams = hparams

        self.input_examples = torch.load(hparams.data_dir + type + '_corpus')

        print('[%s] %d examples is loaded' % (type, len(self.input_examples)))

        self.data_list = []
        for key, value in self.input_examples.items():
            texts = value['texts']
            labels = value['labels']
            dialogues = []
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                dialogues.append({'role': role, 'sentence': sentence})
            self.data_list.append({'labels': labels, 'dialogues': dialogues})

        if vocab_word == None:
            counter = self.build_counter()
            self.vocab_word = self.build_vocab(counter, max_vocab_size)
        else:
            self.vocab_word = vocab_word

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """

        :param index:
        :return:
            Input Examples (vocab_ids)
                dialogues: list of (role, utterance)
                labels: reference summaries for texts
        """
        dialogues = self.data_list[index]['dialogues']
        labels = self.data_list[index]['labels']

        dialogues_ids = [] # (num_turn, seq_len)

        label_tokens = self.tokenize(labels)

        labels_ids = self.tokens2ids(label_tokens, self.vocab_word.token2id,
                                     is_reference=True) #(seq_len)

        for turn_idx, dialogue in enumerate(dialogues):
            # print('turn_idx: ', turn_idx)
            if turn_idx >= self.hparams.max_length:
                break
            tokens = self.tokenize(dialogue['sentence'])

            if len(tokens) >= self.hparams.max_length - 2:
                tokens = tokens[:self.hparams.max_length - 2]

            token_ids = self.tokens2ids(tokens, self.vocab_word.token2id)
            dialogues_ids.append(token_ids)

        padded_dialogues, dialogues_lens = self.pad_sequence(dialogues_ids)

        data = dict()
        data['dialogues'] = dialogues
        data['labels'] = labels
        data['dialogues_ids'] = padded_dialogues
        data['dialogues_lens'] = torch.tensor(dialogues_lens).long()
        data['labels_ids'] = torch.tensor(labels_ids).long()
        return data

    def pad_sequence(self, seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end_idx = lens[i]
            padded_seqs[i, :end_idx] = torch.LongTensor(seq[:end_idx])
        return padded_seqs, lens

    def tokenize(self, sentence):
        return sentence.split()

    def build_counter(self):
        counter = Counter()
        sentence_words = []

        augmented_data_list = [] # add dev vocab
        augmented_data_list.extend(self.data_list)

        dev_examples = torch.load(self.hparams.data_dir + 'dev_corpus')

        print('[%s] %d examples is loaded' % ('Dev', len(dev_examples)))
        for key, value in dev_examples.items():
            texts = value['texts']
            labels = value['labels']
            dialogues = []
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                dialogues.append({'role': role, 'sentence': sentence})
            augmented_data_list.append({'labels': labels, 'dialogues': dialogues})

        for data in augmented_data_list:
            dialogues = data['dialogues']
            for dialogue in dialogues:
                sentence_words.append(self.tokenize(dialogue['sentence']))
            labels = data['labels']
            sentence_words.append(self.tokenize(labels))
        for words in sentence_words:
            counter.update(words)
        return counter

    def build_vocab(self, counter, max_vocab_size):
        print("\n===== Building Word Vocab =========")
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS,
                          '<UNK>': UNK, '<BEGIN>': BEGIN}
        vocab.token2id.update(
            {token: _id + 5 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        print('Vocab size: ', len(vocab.token2id))
        print('==========================================')
        return vocab

    def tokens2ids(self, tokens, token2id, is_reference=False):
        seq = []
        if is_reference is False:
            # Training Corpus
            seq.append(BOS)
            seq.extend([token2id.get(token, UNK) for token in tokens])
            seq.append(EOS)
        else:
            seq.append(BEGIN)
            seq.extend([token2id.get(token, UNK) for token in tokens])
        return seq
