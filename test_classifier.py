import argparse
import os
import imageio
from PIL import Image, ImageOps

import numpy as np
import tensorflow as tf
from keras import backend as K

from data_loader.mnist_data_loader import MNISTDataLoader
from model_trainer_builder import build_model_and_trainer
from utils.config import process_config
from utils.image import normalize_image, denormalize_image

def setup_tf_config():
    tf_sess = tf.Session()
    K.set_session(tf_sess)

def main(file, use_horovod, gpus, config_path, checkpoint):
    config = process_config(config_path, use_horovod, gpus, checkpoint)
    setup_tf_config()

    if os.path.exists(file):
        image = Image.open(file).convert('L')
        image = np.array(ImageOps.invert(image))
        image = np.expand_dims(image, axis=2)
        image = np.expand_dims(image, axis=0)
        image = normalize_image(image)
        
        data_loader = MNISTDataLoader(config)
        model, _ = build_model_and_trainer(config, data_loader)

        prediction = model.predict(image)
        print(np.argmax(np.squeeze(prediction)))
    else:
        raise ValueError('File does not exist')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', type=str)
    ap.add_argument('--horovod', action='store_true', help='use horovod')
    ap.add_argument('--gpus', type=int, default=1, help='number of gpus to use if horovod is disabled')
    ap.add_argument('--config', type=str, default='configs/classifier_config.yml', help='config file to use')
    ap.add_argument('--checkpoint', type=int, default=0, help='checkpoint to continue')

    args = vars(ap.parse_args())

    main(args['file'], args['horovod'], args['gpus'], args['config'], args['checkpoint'])