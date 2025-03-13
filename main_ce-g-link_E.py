"""
This file is the same as "main_glink_ce_blanked.py", but version "E" which
delivered the best results so far.
"""

from pathlib import Path
from main import main
from os import chdir, getcwd
from typing import Literal


MODE: Literal['ckpt', 'test'] = 'ckpt' # 'ckpt' means train (optionally continue from earlier, existing checkpoint; test evaluates)
THIS_DIR = Path(__file__).parent.resolve()
chdir(str(THIS_DIR))
print(getcwd())

DATASET_NAME = "glink"
DATA_DIR = Path(__file__).parent.joinpath('./data').resolve()
DATASET_DIR = DATA_DIR.joinpath(f'./{DATASET_NAME}').resolve()
# Describable Textures Dataset:
DTD_DIR = DATA_DIR.joinpath('./dataset_dtd/images')


main([
                "--gpu", "6",
                "--seed", "0",
                "--test", MODE,
                
                "net",
                "-b", "wideresnet50",
                "-le", "layer2",
                "-le", "layer3",
                "--pretrain_embed_dimension", "1536",
                "--target_embed_dimension", "1536",
                "--patchsize", "3",
                "--meta_epochs", "640",
                "--eval_epochs", "1",
                "--dsc_layers", "2",
                "--dsc_hidden", "1024",
                "--pre_proj", "1",
                "--mining", "1",
                "--noise", "0.015",
                "--radius", "0.75",
                "--p", "0.5",
                "--step", "20",
                "--limit", "392",
                
                "dataset",
                "--distribution", "0", # "1", # 1==judge
                "--mean", "0.5",
                "--std", "0.1",
                "--fg", "1",
                "--rand_aug", "1",
                "--batch_size", "8", # 8
                "--resize", "160", # 288
                "--imagesize", "160", # 288
                "-d", "ce_blanked",
                
                DATASET_NAME,
                str(DATASET_DIR),
                str(DTD_DIR)
            ])