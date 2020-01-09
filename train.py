import argparse
import os
import shutil

import tensorflow as tf
from dotmap import DotMap

from utils.config import process_config
from data_loader.mnist_data_loader import MNISTDataLoader
from model_trainer_builder import build_model_and_trainer

def setup_tf_config(config: DotMap):

    if config.trainer.use_horovod:
        import horovod.keras as hvd
        hvd.init()
        tf.config.experimental.set_visible_devices(str(hvd.local_rank()))
        for device in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    
    is_master = not config.trainer.use_horovod

    if not is_master:
        import horovod.keras as hvd
        is_master = hvd.rank() == 0

    return is_master

def main(use_horovod: bool, gpus: int, config_path: str, checkpoint: int) -> None:
    config = process_config(config_path, use_horovod, gpus, checkpoint)
    
    is_master = setup_tf_config(config)
    
    # copy source files
    if is_master and not os.path.exists(config.exp.source_dir):
        shutil.copytree(
            os.path.abspath(os.path.curdir),
            config.exp.source_dir,
            ignore=lambda src, names: {'datasets', '__pycache__', '.git', 'experiments', 'venv'})
    
    data_loader = MNISTDataLoader(config)

    train_gen = data_loader.get_train_data_generator()

    for _ in range(10):
        x, y = next(train_gen)
        print(x.shape, y.shape)

    _, trainer = build_model_and_trainer(config, data_loader)

    print(f'Start Training Experiment {config.exp.name}')
    trainer.train()


if __name__ == '__main__':
    print(os.path.abspath(os.curdir), os.getpid())

    ap = argparse.ArgumentParser()
    ap.add_argument('--horovod', action='store_true', help='use horovod')
    ap.add_argument('--gpus', type=int, default=1, help='number of gpus to use if horovod is disabled')
    ap.add_argument('--config', type=str, default='configs/config.yml', help='config file to use')
    ap.add_argument('--checkpoint', type=int, default=0, help='checkpoint to continue')

    args = vars(ap.parse_args())

    main(args['horovod'], args['gpus'], args['config'], args['checkpoint'])