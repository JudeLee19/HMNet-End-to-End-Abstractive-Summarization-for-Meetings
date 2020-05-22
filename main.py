import argparse
import os
import logging
import collections

from config.hparams import *
from train import Summarization


def init_logger(path):
  if not os.path.exists(path):
      os.makedirs(path)
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.DEBUG)
  debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
  debug_fh.setLevel(logging.DEBUG)

  info_fh = logging.FileHandler(os.path.join(path, "info.log"))
  info_fh.setLevel(logging.INFO)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
  debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

  ch.setFormatter(info_formatter)
  info_fh.setFormatter(info_formatter)
  debug_fh.setFormatter(debug_formatter)

  logger.addHandler(ch)
  logger.addHandler(debug_fh)
  logger.addHandler(info_fh)

  return logger


def train_model(args):
    hparams = PARAMS
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    summarization = Summarization(hparams)
    summarization.train()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="End-to-End Meeting Summarization (PyTorch)")
    args = arg_parser.parse_args()
    train_model(args)


