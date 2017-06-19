import os

import tensorflow as tf

from glob import glob

from operations import *
from utils import *

class DCGAN (object):
    def __init__(self, session, batch_size=64, input_image_width=108, input_image_height=108, output_image_height=64,
        output_image_width=64, sample_num=64, y_dim=None, z_dim=100, gen_filters_conv1=64, dis_filters_conv1=64,
        gen_units_ful_con_layer=1024, dis_units_ful_con_layer=1024, image_color_dim=3, input_file_pattern='*.jpg',
        dataset='celebA', crop=True):
        
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

        self.dis_bn1 = batch_norm(name='dis_bn1')
        self.dis_bn2 = batch_norm(name='dis_bn2')

        if not self.y_dim:
            self.dis_bn3 = batch_norm(name='dis_bn3')

        self.gen_bn0 = batch_norm(name='gen_bn0')
        self.gen_bn1 = batch_norm(name='gen_bn1')
        self.gen_bn2 = batch_norm(name='gen_bn2')

        if not self.y_dim:
            self.gen_bn3 = batch_norm(name='gen_bn3')

        self.data = glob(os.path.join("./data", self.dataset, self.input_fname_pattern))

        imreadImg = imread(self.data[0]);
        if len(imreadImg.shape) >= 3:   # grayscale or RGB image
            self.image_color_dim = imread(self.data[0]).shape[-1]
        else:
            self.image_color_dim = 1
        self.grayscale = (self.image_color_dim == 1)

        self.create_architecture()

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
            self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
            self.sampler = self.sampler(self.z, self.y)
            self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(inputs)
            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.summation_D = histogram_summary("D", self.D)
        self.summation_D_ = histogram_summary("D_", self.D_)
        self.summation_G = image_summary("G", self.G)

        def cross_entropy_sigmoid_with_logits_mapping():
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


