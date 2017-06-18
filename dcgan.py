import os
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
        if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
            self.image_color_dim = imread(self.data[0]).shape[-1]
        else:
            self.image_color_dim = 1
        self.grayscale = (self.image_color_dim == 1)

        self.create_architecture()

    def create_architecture(self):
        pass


