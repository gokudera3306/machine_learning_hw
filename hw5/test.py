import sys
from argparse import Namespace
from pathlib import Path

import numpy
import tqdm.auto as tqdm
import torch
from matplotlib import pyplot as plt

from fairseq import utils
import logging

from model import build_model, add_transformer_args
from parameter import logger, device, config
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from validate import load_data_iterator, inference_step


def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir) / name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")


def generate_prediction(model, task, sequence_generator, split="test", outfile="./prediction.txt"):
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)

            # do inference
            s, h, r = inference_step(sample, model, task, sequence_generator)

            hyps.extend(h)
            idxs.extend(list(sample['id']))

    # sort based on the order before preprocess
    hyps = [x for _, x in sorted(zip(idxs, hyps))]

    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h + "\n")


task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)

task = TranslationTask.setup_task(task_cfg)

arch_args = Namespace(
    encoder_embed_dim=512,
    encoder_ffn_embed_dim=512,
    encoder_layers=4,
    decoder_embed_dim=512,
    decoder_ffn_embed_dim=1024,
    decoder_layers=4,
    share_decoder_input_output_embed=True,
    dropout=0.3,
)
add_transformer_args(arch_args)
model = build_model(arch_args, task)
model = model.to(device=device)

# pos_emb = model.decoder.embed_positions.weights.cpu().detach()
#
# s = pos_emb.size(dim=0)
#
# final = numpy.zeros([0, s])
# for x in range(0, s):
#     temp = numpy.zeros([0])
#     xx = torch.select(pos_emb, dim=0, index=x)
#     for y in range(0, s):
#         yy = torch.select(pos_emb, dim=0, index=y)
#         rr = torch.nn.functional.cosine_similarity(xx, yy, dim=0)
#         temp = numpy.append(temp, rr)
#     final = numpy.append(final, numpy.reshape(temp, (-1, s)), axis=0)
#
# plt.imshow(final, cmap='hot', interpolation='nearest')
# plt.show()

sequence_generator = task.build_generator([model], config)

# checkpoint_last.pt : latest epoch
# checkpoint_best.pt : highest validation bleu
# avg_last_5_checkpoint.pt: the average of last 5 epochs
try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
generate_prediction(model, task, sequence_generator)
