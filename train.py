import os
import logging

from datetime import datetime
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import AMIDataset
from models.model import SummarizationModel

class Summarization(object):
    def __init__(self, hparams):
        self.hparams = hparams
        print('self.hparams:', self.hparams)
        self.logger = logging.getLogger(__name__)
        if hparams.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def build_dataloader(self):
        self.train_dataset = AMIDataset(self.hparams, type='train')
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True
        )
        self.vocab_word = self.train_dataset.vocab_word

    print("""
           # -------------------------------------------------------------------------
           #   DATALOADER FINISHED
           # -------------------------------------------------------------------------
           """)

    def build_model(self):
        # Define model
        self.model = SummarizationModel(self.hparams, self.vocab_word)
        # Multi-GPU

        # Define Loss and Optimizer

    def setup_training(self):
        pass

    def train(self):
        self.build_dataloader()
        self.build_model()

        for epoch in range(self.hparams.num_epochs):
            tqdm_batch_iterator = tqdm(self.train_dataloader)
            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                data = batch
                dialogues_ids = data['dialogues_ids']
                labels_ids = data['labels_ids']

                print('shape(dialogues_ids): ', dialogues_ids.shape)
                print('shape(labels_ids): ', labels_ids.shape)
                print('\n')
                self.model(dialogues_ids, labels_ids)

                break
            break
