import argparse
import os
import shutil

import tensorflow as tf
from dotmap import DotMap
from keras import backend as K

from utils.config import process_config

def main(use_horovod: bool, gpus: int, config_path: str, checkpoint: int) -> None:
    config = process_config(config_path, use_horovod, gpus, checkpoint)
    print(config)

if __name__ == '__main__':
    print(os.path.abspath(os.curdir), os.getpid())

    ap = argparse.ArgumentParser()
    ap.add_argument('--horovod', action='store_true', help='use horovod')
    ap.add_argument('--gpus', type=int, default=1, help='number of gpus to use if horovod is disabled')
    ap.add_argument('--config', type=str, default='configs/config.yml', help='config file to use')
    ap.add_argument('--checkpoint', type=int, default=0, help='checkpoint to continue')

    args = vars(ap.parse_args())

    main(args['horovod'], args['gpus'], args['config'], args['checkpoint'])