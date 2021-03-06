import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils

import tensorflow as tf

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--mode", default='test',
                    help='mode to run [train/test/reg/vizu]')
parser.add_argument("--exp", default='mnist',
                    help='dataset [mnist/celebA/dsprites].'\
                    ' celebA/dsprites Not implemented yet')
parser.add_argument("--method",
                    help='algo to train [swae/vae]')
parser.add_argument("--work_dir")
parser.add_argument("--weights_file")

FLAGS = parser.parse_args()

def main():

    # Select dataset to use
    if FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.exp == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'grassli':
        opts = configs.config_grassli
    elif FLAGS.exp == 'grassli_small':
        opts = configs.config_grassli_small
    else:
        assert False, 'Unknown experiment configuration'

    # Select training method
    if FLAGS.method:
        opts['method'] = FLAGS.method

    # Working directory
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir

    # Verbose
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Create directories
    utils.create_dir(opts['method'])
    work_dir = os.path.join(opts['method'],opts['work_dir'])
    utils.create_dir(work_dir)
    utils.create_dir(os.path.join(work_dir, 'checkpoints'))

    # Dumping all the configs to the text file
    with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    #Reset tf graph
    tf.reset_default_graph()

    # build WAE
    wae = WAE(opts)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        wae.train(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="test":
        wae.test(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="reg":
        wae.reg(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="vizu":
        wae.vizu(data, opts['work_dir'], FLAGS.weights_file)

main()
