# -*- coding = utf-8 -*-
# @time:2023/9/25
# Author: YuantaoChen
from cmath import tau
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pypose as pp

import network
from dataloaders import dataloader
import opts
import rpmg
from interpolation import Interpolation1d


if __name__ == '__main__':
    hparams = opts.get_opts_base().parse_args()
    exp_folder = Path(hparams.exp_name)
    if not exp_folder.exists():
        os.makedirs(exp_folder.absolute())
    exp_name = str(max([int(name.name) for name in exp_folder.iterdir() if str.isdigit(name.name)] + [0]) + 1)
    writer = SummaryWriter(exp_folder / exp_name / 'tb')
    logging.basicConfig(level=logging.DEBUG
                        , format="[%(levelname)s] %(asctime)-9s - %(filename)-8s:%(lineno)s line - %(message)s"
                        , datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = dataloader.load_dataset(hparams, split='train')
    val_dataset = dataloader.load_dataset(hparams, split='val')
    test_dataset = dataloader.load_dataset(hparams, split='train')

    interp = Interpolation1d(test_dataset, 'linear')
    output_dir = exp_folder / exp_name / 'eval' 
    os.makedirs(output_dir.absolute(), exist_ok=True)
    for batch in test_dataset.dataset:
        timestamps = batch['timestamp'].reshape(-1, 1).to(device)
        out_x, out_q = interp.get_output(timestamps)
        out_mat = pp.SE3(torch.cat([out_x, out_q], dim=-1)).matrix()
        for i in range(len(timestamps)):
            ts = float(timestamps[i]) + hparams.start_timestamp
            mat = out_mat[i].cpu().numpy()
            np.savetxt(output_dir / f'{ts}.txt', mat)
    logger.info(f'evaluation results saved to {output_dir}')

