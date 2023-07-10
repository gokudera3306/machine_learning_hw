import random
import sys
from argparse import Namespace
from pathlib import Path
import logging

import numpy as np
import torch

data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
urls = (
    "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.data.tgz",
    "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.test.tgz"
)
file_names = (
    'ted2020.tgz',  # train & dev
    'test.tgz',  # test
)
prefix = Path(data_dir).absolute() / dataset_name

"""## Language"""

src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'

config = Namespace(
    datadir="./DATA/data-bin/ted2020",
    savedir="./checkpoints/rnn",
    source_lang=src_lang,
    target_lang=tgt_lang,

    # cpu threads when fetching & processing data.
    num_workers=8,
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,

    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,

    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,

    # maximum epochs for training
    max_epoch=40,
    start_epoch=1,

    # beam size for beam search
    beam=9,
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2,
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process="sentencepiece",

    # checkpoints
    keep_last_epochs=5,
    resume=None,  # if resume from checkpoint name (under config.savedir)
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO", # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)

proj = "hw5.seq2seq"
logger = logging.getLogger(proj)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

seed = 33
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
