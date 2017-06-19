from __future__ import division
from six.moves import xrange
from glob import glob

import os
import time
import re
import math
import logging

import tensorflow as tf
import numpy as np

from operations import *
from utils import *

logging.basicConfig(filename='./dcgan.log', filemode="a+", format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p')

def convolution_same_size_output(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN (object):
    def __init__(self, session, batch_size=64, input_image_width=108, input_image_height=108, output_image_height=64,
        output_image_width=64, sample_num=64, y_dim=None, z_dim=100, gen_filters_conv1=64, dis_filters_conv1=64,
        gen_units_ful_con_layer=1024, dis_units_ful_con_layer=1024, image_color_dim=3, input_file_pattern='*.jpg',
        dataset='celebA', crop=True, checkpoint_directory=None, sample_directory=None):
        
        self.session = session
        self.crop = crop
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width
        self.output_image_height = output_image_height
        self.output_image_width = output_image_width
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.input_file_pattern = input_file_pattern

        self.gen_filters_conv1 = gen_filters_conv1
        self.dis_filters_conv1 = dis_filters_conv1
        self.gen_units_ful_con_layer = gen_units_ful_con_layer
        self.dis_units_ful_con_layer = dis_units_ful_con_layer

        self.dis_bn1 = Batch_Normalization(name='dis_bn1')
        self.dis_bn2 = Batch_Normalization(name='dis_bn2')

        if not self.y_dim:
            self.dis_bn3 = Batch_Normalization(name='dis_bn3')

        self.gen_bn0 = Batch_Normalization(name='gen_bn0')
        self.gen_bn1 = Batch_Normalization(name='gen_bn1')
        self.gen_bn2 = Batch_Normalization(name='gen_bn2')

        if not self.y_dim:
            self.gen_bn3 = Batch_Normalization(name='gen_bn3')

        self.dataset = dataset
        # Server data path
        self.data = glob(os.path.join("/Neutron9/yash.lal/data", self.dataset, self.input_file_pattern))
        # self.data = glob(os.path.join("./data", self.dataset, self.input_file_pattern))
        self.checkpoint_directory = checkpoint_directory

        imreadImg = imread(self.data[0]);
        if len(imreadImg.shape) >= 3:   # grayscale or RGB image
            self.image_color_dim = imread(self.data[0]).shape[-1]
        else:
            self.image_color_dim = 1
        self.grayscale = (self.image_color_dim == 1)

        self.create_architecture()

    @property
    def model_directory(self):
        return "{}_{}_{}_{}".format(self.dataset, self.batch_size, self.output_image_height, self.output_image_width)

    def save(self, checkpoint_directory, step):
        model_name = "celebA-dcgan.model"
        checkpoint_directory = os.path.join(checkpoint_directory, self.model_directory)

        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        self.saver.save(self.session, os.path.join(checkpoint_directory, model_name), global_step=step)

    def load(self, checkpoint_directory):
        logging.debug("Reading checkpoints")
        checkpoint_directory = os.path.join(checkpoint_directory, self.model_directory)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_directory)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_directory, checkpoint_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",checkpoint_name)).group(0))
            logging.debug("Success finding {}".format(checkpoint_name))
            return True, counter
        else:
            logging.debug("Failed to find checkpoint")
            return False, 0

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if not self.y_dim:
                h0 = leaky_relu(conv2d(image, self.dis_filters_conv1, name='dis_h0_conv'))
                h1 = leaky_relu(self.dis_bn1(conv2d(h0, self.dis_filters_conv1*2, name='dis_h1_conv')))
                h2 = leaky_relu(self.dis_bn2(conv2d(h1, self.dis_filters_conv1*4, name='dis_h2_conv')))
                h3 = leaky_relu(self.dis_bn3(conv2d(h2, self.dis_filters_conv1*8, name='dis_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')
                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = concatenate_conditioning_vector(image, yb)
                h0 = leaky_relu(conv2d(x, self.image_color_dim + self.y_dim, name='dis_h0_conv'))
                h0 = concatenate_conditioning_vector(h0, yb)
                h1 = leaky_relu(self.dis_bn1(conv2d(h0, self.dis_filters_conv1 + self.y_dim, name='dis_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])      
                h1 = concatenate([h1, y], 1)
                h2 = leaky_relu(self.dis_bn2(linear(h1, self.dis_units_ful_con_layer, 'dis_h2_lin')))
                h2 = concatenate([h2, y], 1)
                h3 = linear(h2, 1, 'dis_h3_lin')
                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h, s_w = self.output_image_height, self.output_image_width
                s_h2, s_w2 = convolution_same_size_output(s_h, 2), convolution_same_size_output(s_w, 2)
                s_h4, s_w4 = convolution_same_size_output(s_h2, 2), convolution_same_size_output(s_w2, 2)
                s_h8, s_w8 = convolution_same_size_output(s_h4, 2), convolution_same_size_output(s_w4, 2)
                s_h16, s_w16 = convolution_same_size_output(s_h8, 2), convolution_same_size_output(s_w8, 2)

                # projection of z
                self.z_, self.h0_w, self.h0_b = linear(z, self.gen_filters_conv1*8*s_h16*s_w16, 'gen_h0_lin', with_w=True)

                # reshape
                self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gen_filters_conv1 * 8])
                h0 = tf.nn.relu(self.gen_bn0(self.h0))
                self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gen_filters_conv1*4], name='gen_h1', with_w=True)
                h1 = tf.nn.relu(self.gen_bn1(self.h1))
                h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gen_filters_conv1*2], name='gen_h2', with_w=True)
                h2 = tf.nn.relu(self.gen_bn2(h2))
                h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gen_filters_conv1*1], name='gen_h3', with_w=True)
                h3 = tf.nn.relu(self.gen_bn3(h3))
                h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.image_color_dim], name='gen_h4', with_w=True)
                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_image_height, self.output_image_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concatenate([z, y], 1)
                h0 = tf.nn.relu(self.gen_bn0(linear(z, self.gen_units_ful_con_layer, 'gen_h0_lin')))
                h0 = concatenate([h0, y], 1)
                h1 = tf.nn.relu(self.gen_bn1(linear(h0, self.gen_filters_conv1*2*s_h4*s_w4, 'gen_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gen_filters_conv1 * 2])
                h1 = concatenate_conditioning_vector(h1, yb)
                h2 = tf.nn.relu(self.gen_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gen_filters_conv1 * 2], name='gen_h2')))
                h2 = concatenate_conditioning_vector(h2, yb)
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.image_color_dim], name='gen_h3'))

    def create_architecture(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        if self.crop:
            image_dimensions = [self.output_image_height, self.output_image_width, self.image_color_dim]
        else:
            image_dimensions = [self.input_image_height, self.input_image_width, self.image_color_dim]

        self.real_images = tf.placeholder(tf.float32, [self.batch_size] + image_dimensions, name='real_images')
        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dimensions, name='sample_inputs')

        real_images = self.real_images
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.summation_z = histogram_summary('z', self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits = self.discriminator(real_images, self.y, reuse=False)
            self.sampler = self.sampler(self.z, self.y)
            self.D, self.D_logits = self.discriminator(real_images, self.y, reuse=False)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(real_images)
            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.summation_D = histogram_summary("D", self.D)
        self.summation_D_ = histogram_summary("D_", self.D_)
        self.summation_G = image_summary("G", self.G)

        def cross_entropy_sigmoid_with_logits_mapping(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.dis_loss_real = tf.reduce_mean(cross_entropy_sigmoid_with_logits_mapping(self.D_logits, tf.ones_like(self.D)))
        self.dis_loss_fake = tf.reduce_mean(cross_entropy_sigmoid_with_logits_mapping(self.D_logits_, tf.zeros_like(self.D_)))
        self.gen_loss = tf.reduce_mean(cross_entropy_sigmoid_with_logits_mapping(self.D_logits_, tf.ones_like(self.D_)))
        self.dis_loss_real_sum = scalar_summary("dis_loss_real", self.dis_loss_real)
        self.dis_loss_fake_sum = scalar_summary("dis_loss_fake", self.dis_loss_fake)                 
        self.dis_loss = self.dis_loss_real + self.dis_loss_fake
        self.summation_gen_loss = scalar_summary("gen_loss", self.gen_loss)
        self.summation_dis_loss = scalar_summary("dis_loss", self.dis_loss)

        trainable_variables = tf.trainable_variables()
        self.dis_vars = [var for var in trainable_variables if 'dis_' in var.name]
        self.gen_vars = [var for var in trainable_variables if 'gen_' in var.name]
        self.saver = tf.train.Saver()

    def train(self, config):
        dis_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.dis_loss, var_list=self.dis_vars)
        gen_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.gen_loss, var_list=self.gen_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.summation_G = merge_summary([self.summation_z, self.summation_D_, self.summation_G, self.dis_loss_fake_sum, self.summation_gen_loss])
        self.summation_D = merge_summary([self.summation_z, self.summation_D, self.dis_loss_real_sum, self.summation_dis_loss])
        self.summary_writer = SummaryWriter("./summary", self.session.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
        sample_files = self.data[0:self.sample_num]
        sample = [get_image(sample_file, input_height=self.input_image_height, input_width=self.input_image_width, resize_height=self.output_image_height,
                    resize_width=self.output_image_width, crop=self.crop, grayscale=self.grayscale) for sample_file in sample_files]
        if (self.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        load_success, checkpoint_counter = self.load(self.checkpoint_directory)
        if load_success:
            counter = checkpoint_counter
            logging.debug("Loaded checkpoint")
        else:
            logging.debug("Failed in loading")

        for epoch in xrange(config.epoch):
            # Server data path
            self.data = glob(os.path.join("/Neutron9/yash.lal/data", self.dataset, self.input_file_pattern))
            # self.data = glob(os.path.join("./data", config.dataset, self.input_file_pattern))
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, input_height=self.input_image_height, input_width=self.input_image_width, resize_height=self.output_image_height,
                    resize_width=self.output_image_width, crop=self.crop, grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

            # Discriminator update
            _, summary_string = self.session.run([dis_optimizer, self.summation_D], feed_dict={ self.inputs: batch_images, self.z: batch_z })
            self.summary_writer.add_summary(summary_string, counter)

            # Generator update
            _, summary_string = self.session.run([gen_optimizer, self.summation_G], feed_dict={ self.z: batch_z })
            self.summary_writer.add_summary(summary_string, counter)

            # Test - make sure discriminator loss does not go to 0
            # _, summary_string = self.session.run([gen_optimizer, self.summation_G], feed_dict={ self.z: batch_z })
            # self.summary_writer.add_summary(summary_string, counter)

            errD_fake = self.dis_loss_fake.eval({ self.z: batch_z })
            errD_real = self.dis_loss_real.eval({ self.inputs: batch_images })
            errG = self.gen_loss.eval({self.z: batch_z})

            counter += 1
            logging.debug("Epoch: [%2d] [%4d/%4d] time: %4.4f, dis_loss: %.8f, gen_loss: %.8f" % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
            
            if np.mod(counter, 100) == 1:
                try:
                    samples, dis_loss, gen_loss = self.session.run([self.sampler, self.dis_loss, self.gen_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_inputs,
                        },
                    )
                    manifold_height = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_weight = int(np.floor(np.sqrt(samples.shape[0])))
                    save_images(samples, [manifold_height, manifold_weight],
                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_directory, epoch, idx))
                    logging.debug("[Sample] dis_loss: %.8f, gen_loss: %.8f" % (dis_loss, gen_loss)) 
                except:
                    logging.debug("Error in one picture")

            if np.mod(counter, 500) == 2:
                self.save(config.checkpoint_directory, counter)

    # similar to generator
    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_image_height, self.output_image_width
                s_h2, s_w2 = convolution_same_size_output(s_h, 2), convolution_same_size_output(s_w, 2)
                s_h4, s_w4 = convolution_same_size_output(s_h2, 2), convolution_same_size_output(s_w2, 2)
                s_h8, s_w8 = convolution_same_size_output(s_h4, 2), convolution_same_size_output(s_w4, 2)
                s_h16, s_w16 = convolution_same_size_output(s_h8, 2), convolution_same_size_output(s_w8, 2)

                # projection of z and reshaping
                h0 = tf.reshape(linear(z, self.gen_filters_conv1*8*s_h16*s_w16, 'gen_h0_lin'), [-1, s_h16, s_w16, self.gen_filters_conv1 * 8])
                h0 = tf.nn.relu(self.gen_bn0(h0, train=False))
                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gen_filters_conv1*4], name='gen_h1')
                h1 = tf.nn.relu(self.gen_bn1(h1, train=False))
                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gen_filters_conv1*2], name='gen_h2')
                h2 = tf.nn.relu(self.gen_bn2(h2, train=False))
                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gen_filters_conv1*1], name='gen_h3')
                h3 = tf.nn.relu(self.gen_bn3(h3, train=False))
                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.image_color_dim], name='gen_h4')
                return tf.nn.tanh(h4)

            else:
                s_h, s_w = self.output_image_height, self.output_image_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concatenate([z, y], 1)
                h0 = tf.nn.relu(self.gen_bn0(linear(z, self.gen_units_ful_con_layer, 'gen_h0_lin'), train=False))
                h0 = concatenate([h0, y], 1)
                h1 = tf.nn.relu(self.gen_bn1(linear(h0, self.gen_filters_conv1*2*s_h4*s_w4, 'gen_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gen_filters_conv1 * 2])
                h1 = concatenate_conditioning_vector(h1, yb)
                h2 = tf.nn.relu(self.gen_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gen_filters_conv1 * 2], name='gen_h2'), train=False))
                h2 = concatenate_conditioning_vector(h2, yb)
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.image_color_dim], name='gen_h3'))


