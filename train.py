import argparse
import os
import shutil

import tensorflow as tf
from dotmap import DotMap
from keras import backend as K

from utils.config import process_config

def setup_tf_session(config: DotMap):
    tf_config = tf.ConfigProto()

    if config.trainer.use_horovod:
        import horovod.keras as hvd
        hvd.init()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.visible_device_list = str(hvd.local_rank())
    
    is_master = not config.trainer.use_horovod

    if not is_master:
        import horovod.keras as hvd
        is_master = hvd.rank() == 0
    
    tf_sess = tf.Session(config=tf_config)
    K.set_session(tf_sess)

    return is_master

def main(use_horovod: bool, gpus: int, config_path: str, checkpoint: int) -> None:
    config = process_config(config_path, use_horovod, gpus, checkpoint)
    
    is_master = setup_tf_session(config)
    
    # copy source files
    if is_master and not os.path.exists(config.exp.source_dir):
        shutil.copytree(
            os.path.abspath(os.path.curdir),
            config.exp.source_dir,
            ignore=lambda src, names: {'datasets', '__pycache__', '.git', 'experiments', 'venv'})
    
    

if __name__ == '__main__':
    print(os.path.abspath(os.curdir), os.getpid())

    ap = argparse.ArgumentParser()
    ap.add_argument('--horovod', action='store_true', help='use horovod')
    ap.add_argument('--gpus', type=int, default=1, help='number of gpus to use if horovod is disabled')
    ap.add_argument('--config', type=str, default='configs/config.yml', help='config file to use')
    ap.add_argument('--checkpoint', type=int, default=0, help='checkpoint to continue')

    args = vars(ap.parse_args())

    main(args['horovod'], args['gpus'], args['config'], args['checkpoint'])