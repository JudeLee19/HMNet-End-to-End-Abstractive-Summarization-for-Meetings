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
from utils.checkpointing import CheckpointManager, load_checkpoint
from predictor import Predictor


class Summarization(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self._logger = logging.getLogger(__name__)
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

        self.test_dataset = AMIDataset(self.hparams, type='test', vocab_word=self.vocab_word)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            drop_last=False
        )

    print("""
           # -------------------------------------------------------------------------
           #   DATALOADER FINISHED
           # -------------------------------------------------------------------------
           """)

    def build_model(self):
        # Define model
        self.model = SummarizationModel(self.hparams, self.vocab_word)

        # Multi-GPU
        self.model = self.model.to(self.device)

        # Use Multi-GPUs
        if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

        # Define Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.optimizer_adam_beta1,
                                                                                               self.hparams.optimizer_adam_beta2))

        # Define predictor
        self.predictor = Predictor(self.hparams, model=None, vocabs=self.vocab_word,
                                   checkpoint=self.hparams.load_pthpath)

    def setup_training(self):
        self.save_dirpath = self.hparams.save_dirpath
        self.summary_writer = SummaryWriter(self.save_dirpath)
        self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath, hparams=self.hparams)

        # If loading from checkpoint, adjust start epoch and load parameters.
        if self.hparams.load_pthpath == "":
            self.start_epoch = 1
        else:
            # "path/to/checkpoint_xx.pth" -> xx
            self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
            self.start_epoch += 1
            model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.previous_model_path = self.hparams.load_pthpath
            print("Loaded model from {}".format(self.hparams.load_pthpath))

        print(
            """
            # -------------------------------------------------------------------------
            #   Setup Training Finished
            # -------------------------------------------------------------------------
            """
        )

    def train(self):
        self.build_dataloader()
        self.build_model()
        self.setup_training()

        train_begin = datetime.utcnow()  # News
        global_iteration_step = 0
        for epoch in range(self.hparams.num_epochs):

            self.evaluate()
            tqdm_batch_iterator = tqdm(self.train_dataloader)
            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                data = batch
                dialogues_ids = data['dialogues_ids'].to(self.device)
                labels_ids = data['labels_ids'].to(self.device) # [batch, tgt_seq_len]
                src_masks = data['src_masks'].to(self.device)

                # print('batch_idx: ', batch_idx)
                # print('shape(dialogues_ids): ', dialogues_ids.shape)
                # print('shape(labels_ids): ', labels_ids.shape)
                # print('shape(src_masks): ', src_masks.shape)
                # print('\n')

                logits = self.model(inputs=dialogues_ids, targets=labels_ids,
                                    src_masks=src_masks) # [batch x tgt_seq_len, vocab_size]

                labels_ids = labels_ids.view(labels_ids.shape[0] * labels_ids.shape[1]) # [batch x tgt_seq_len]

                loss = self.criterion(logits, labels_ids)
                loss.backward()

                self.optimizer.step()
                # gradient cliping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
                self.optimizer.zero_grad()

                global_iteration_step += 1
                description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                    datetime.utcnow() - train_begin,
                    epoch,
                    global_iteration_step, loss,
                    self.optimizer.param_groups[0]['lr'])
                tqdm_batch_iterator.set_description(description)

            # -------------------------------------------------------------------------
            #   ON EPOCH END  (checkpointing and validation)
            # -------------------------------------------------------------------------
            self.checkpoint_manager.step(epoch)
            self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
            self._logger.info(self.previous_model_path)

            torch.cuda.empty_cache()

            # -------------------------------------------------------------------------
            #   Evaluation
            # -------------------------------------------------------------------------
            if epoch >= 5:
                self.evaluate()

    def evaluate(self):
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
                data = batch
                dialogues_ids = data['dialogues_ids'].to(self.device)
                labels_ids = data['labels_ids'].to(self.device)  # [batch, tgt_seq_len]
                src_masks = data['src_masks'].to(self.device)

                summaries = self.predictor.inference(inputs=dialogues_ids, src_masks=src_masks)
