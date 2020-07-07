import argparse
import os
import logging
import collections

from config.hparams import *
from train import Summarization
import torch



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

    save_path = args.save_path
    if save_path == '':
        raise ValueError("Muse provide save path !")

    hparams = hparams._replace(save_dirpath=save_path)
    hparams = hparams._replace(use_role=args.use_role)
    hparams = hparams._replace(use_role=args.use_pos)

    seed_value = 666
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

    if len(hparams.gpu_ids) > 0:
        torch.cuda.set_device(hparams.gpu_ids[0])
        torch.cuda.manual_seed(seed_value)

    print('hparams.save_dirpath: ', hparams.save_dirpath)
    summarization = Summarization(hparams, mode='train')
    summarization.train()


def evaluate_model(args):
    hparams = PARAMS
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

    model_path = args.model_path
    if model_path == '':
        raise ValueError('Must provide model_path !')
    save_dirpath =  '/'.join(model_path.split('/')[:-1])
    save_dirpath = save_dirpath + '/'
    hparams = hparams._replace(save_dirpath=save_dirpath)

    # gen_max_length
    gen_max_length = args.gen_max_length
    print('gen_max_length: ', gen_max_length)
    hparams = hparams._replace(gen_max_length=gen_max_length)
    hparams = hparams._replace(use_role=args.use_role)
    hparams = hparams._replace(use_role=args.use_pos)

    epoch = hparams.start_eval_epoch
    print('\n ========= [Evaluation Start Epoch: ', epoch, ']================== ')
    for i in range(int(epoch), 100):
        load_pthpath = '/'.join(model_path.split('/')[:-1]) + '/checkpoint_' + str(i) + '.pth'
        hparams= hparams._replace(load_pthpath=load_pthpath)
        print('hparams.load_pthpath: ', hparams.load_pthpath)
        summarization = Summarization(hparams, mode='eval')
        summarization.evaluate(epoch=i)
        del summarization
    print('\n')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="End-to-End Meeting Summarization (PyTorch)")
    arg_parser.add_argument("--mode", dest="mode", type=str, default="",
                            help="(train/eval)")
    arg_parser.add_argument("--model_path", dest="model_path", type=str, default="",
                            help="trained model path")
    arg_parser.add_argument("--save_path", dest="save_path", type=str, default="",
                            help="path to save the trained model")
    arg_parser.add_argument("--gen_max_length", dest="gen_max_length", type=int,
                            default=500, help="gen_max_length")
    arg_parser.add_argument("--use_role", dest="use_role", type=bool,
                            default=False)
    arg_parser.add_argument("--use_pos", dest="use_pos", type=bool,
                            default=False)


    args = arg_parser.parse_args()
    mode = args.mode

    if mode == 'train':
        train_model(args)
    elif mode == 'eval':
        evaluate_model(args)


