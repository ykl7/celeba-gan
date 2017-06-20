import os
import scipy.misc
import logging

import numpy as np
import tensorflow as tf

from dcgan import DCGAN
from utils import pp, show_all_variables, visualize

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Number of epochs to train for")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for Adam optimizer [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam optimizer [0.5]")
flags.DEFINE_integer("train_size", np.inf, "Train images size [np.inf]")
flags.DEFINE_integer("batch_size", 64, "Batch images size [64]")
flags.DEFINE_integer("input_image_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_image_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_image_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_image_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA]")
flags.DEFINE_string("input_file_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_directory", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_directory", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    logging.basicConfig(filename='./run.log', filemode="a+", format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p')
    logging.debug(pp.pprint(flags.FLAGS.__flags))
    if FLAGS.input_image_width is None:
        FLAGS.input_image_width = FLAGS.input_image_height
    if FLAGS.output_image_width is None:
        FLAGS.output_image_width = FLAGS.output_image_height
    if not os.path.exists(FLAGS.checkpoint_directory):
        os.makedirs(FLAGS.checkpoint_directory)
    if not os.path.exists(FLAGS.sample_directory):
        os.makedirs(FLAGS.sample_directory)

    run_configuration = tf.ConfigProto()
    run_configuration.gpu_options.allow_growth=True

    with tf.Session(config=run_configuration) as session:
        dcgan = DCGAN(session, input_image_width=FLAGS.input_image_width, input_image_height=FLAGS.input_image_height, output_image_width=FLAGS.output_image_width,
            output_image_height=FLAGS.output_image_height, batch_size=FLAGS.batch_size, sample_num=FLAGS.batch_size, dataset=FLAGS.dataset,
            input_file_pattern=FLAGS.input_file_pattern, crop=FLAGS.crop, checkpoint_directory=FLAGS.checkpoint_directory, sample_directory=FLAGS.sample_directory)

        show_all_variables()
        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_directory)[0]:
                raise Exception("No pre-trained model exists")

        # visualization activation for Alec Radford's utils.py function - does not work since not implemented (and no idea how to)
        OPTION = 1
        visualize(session, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()

